#pragma once

#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "otter/core/dtype.h"
#include "otter/detail/stride_utils.h"
#include "otter/kernel/backend.h"  // full def needed for from_data template body
#include "otter/memory/buffer.h"   // full def needed for from_data template body

namespace otter {

// Forward declarations — full definitions follow after Tensor or in other headers.
struct GradAccumulator;  // defined below, after Tensor (needs Tensor to be complete)
class  Operation;        // defined in include/otter/ops/operation.h (added in step 2)

// Tensor — value-type multi-dimensional array.
//
// Multiple Tensors may share one Buffer (view semantics): copies are cheap,
// each Tensor has independent shape/stride/offset metadata.
//
// The Rule of Zero applies: shared_ptr<Buffer> manages the allocation; all
// five special members are compiler-generated and correct.
//
// Autograd: requires_grad_, is_leaf_, grad_accum_, grad_op_ implement reverse-mode
// automatic differentiation. execute() in Operation sets these on outputs; the user
// sets requires_grad via the factory methods (zeros/from_data with requires_grad=true).
class Tensor {
public:
    // ── Factory ──────────────────────────────────────────────────────────────

    // Allocates a zero-initialised contiguous tensor via backend.
    // requires_grad=true: tensor participates in backward(); grad_accum_ is allocated.
    static Tensor zeros(const std::vector<std::size_t>& shape,
                        Backend& backend,
                        DType dtype       = DType::Float64,
                        bool  requires_grad = false);
    static Tensor zeros(const std::vector<std::size_t>&, Backend&&,
                        DType = DType::Float64, bool = false) = delete;  // Backend must outlive Tensor

    // Copies data into a fresh contiguous tensor.
    // T must have a dtype_utils::dtype_of<T> specialisation (currently double).
    template<typename T>
    static Tensor from_data(const std::vector<T>&        data,
                            const std::vector<std::size_t>& shape,
                            Backend& backend,
                            bool     requires_grad = false);
    template<typename T>
    static Tensor from_data(const std::vector<T>&,
                            const std::vector<std::size_t>&,
                            Backend&&, bool = false) = delete;  // Backend must outlive Tensor

    // ── Rule of Zero — shared_ptr<Buffer> manages the allocation ─────────────
    Tensor() = default;

    // ── Metadata ─────────────────────────────────────────────────────────────

    [[nodiscard]] bool defined() const noexcept { return buffer_ != nullptr; }

    [[nodiscard]] const std::vector<std::size_t>& shape()  const noexcept { return shape_;  }
    [[nodiscard]] const std::vector<std::size_t>& stride() const noexcept { return stride_; }
    [[nodiscard]] std::size_t                     offset() const noexcept { return offset_; }
    [[nodiscard]] DType                           dtype()  const noexcept { return dtype_;  }
    [[nodiscard]] bool is_contiguous()  const noexcept { return is_contiguous_; }
    [[nodiscard]] std::size_t numel()   const noexcept;
    // Returns the backend this tensor was allocated on.
    // Returns a mutable Backend& from a const method: Backend is a non-owning
    // observer of the tensor, not owned state. Returning const Backend& would
    // prevent allocating tensors from const Tensor refs in backward().
    [[nodiscard]] Backend& backend() const;  // asserts defined()

    // ── Autograd metadata ─────────────────────────────────────────────────────
    [[nodiscard]] bool requires_grad() const noexcept { return requires_grad_; }

    // True for tensors created directly by the user (via zeros/from_data).
    // False for tensors produced by Operation::execute().
    // Distinct from grad_op_==nullptr, which can also mean the graph was cleared.
    [[nodiscard]] bool is_leaf()       const noexcept { return is_leaf_;       }

    // ── Buffer access (internal — for KernelEngine only) ─────────────────────
    // Raw data is further gated by Passkey<KernelEngine> inside Buffer.
    [[nodiscard]] Buffer&       mutable_buffer();
    [[nodiscard]] const Buffer& buffer() const;

    // ── Layout ───────────────────────────────────────────────────────────────

    // Returns a new Tensor sharing the same Buffer with new shape/stride.
    // Buffer aliasing: mutations via one view are visible in all others sharing it.
    // No autograd — pure layout alias. The returned Tensor has no grad history.
    // Precondition: new_shape.size() == new_stride.size().
    [[nodiscard]] Tensor view(std::vector<std::size_t> new_shape,
                              std::vector<std::size_t> new_stride) const;

    // Returns a contiguous copy if !is_contiguous(), otherwise *this.
    // Intentionally not differentiable — it is a data normalisation step.
    [[nodiscard]] Tensor contiguous() const;

    // ── Scalar read (device-safe) ─────────────────────────────────────────────
    // Uses dispatch_element_read — makes device→host transfer explicit.
    // On a CUDA backend this becomes a kernel launch; never a silent host deref.
    [[nodiscard]] double at(std::initializer_list<std::size_t> indices) const;

    // ── In-place fill ────────────────────────────────────────────────────────
    // Writes value to every element. Requires: (1) contiguous, (2) uniquely
    // owned buffer (use_count == 1). Both conditions hold for freshly allocated
    // tensors from zeros(). Not safe on views or copies. Not differentiable.
    void fill_(double value);

    // ── Operations ───────────────────────────────────────────────────────────
    // Each creates the corresponding Operation and calls execute().
    // mul() and matmul() are added in steps 4 and 5 respectively.

    [[nodiscard]] Tensor add(const Tensor& other) const;
    [[nodiscard]] Tensor mul(const Tensor& other) const;
    [[nodiscard]] Tensor sum() const;

    // ── Autograd methods ──────────────────────────────────────────────────────

    // Returns the accumulated gradient tensor.
    // Undefined (default-constructed) if no gradient has been accumulated yet.
    [[nodiscard]] Tensor grad() const noexcept;

    // Add `incoming` into this tensor's gradient accumulator.
    // incoming is detached internally to prevent second-order grad cycles.
    // Precondition: requires_grad() == true.
    // Implemented alongside Tensor::add() (step 3) — declared here for the interface.
    void accumulate_grad(const Tensor& incoming) const;

    // Reset the gradient accumulator to undefined.
    // Call before each forward pass in a training loop.
    void zero_grad() noexcept;

    // Returns a shallow copy sharing the same Buffer with requires_grad=false,
    // no grad_accum_, no grad_op_. No data is copied.
    // Use to stop gradient flow or pass data to non-differentiable operations.
    // Note: is_leaf_ is NOT changed — detach does not make a computed tensor a leaf.
    [[nodiscard]] Tensor detach() const noexcept;

    // Entry point for backward pass. Seeds gradient with ones if not provided.
    // retain_graph=false (default): clears saved inputs after traversal.
    // retain_graph=true: leaves graph intact for a second backward call.
    // Implemented in step 3 alongside Operation and the concrete ops.
    void backward(bool retain_graph = false);
    void backward(Tensor seed, bool retain_graph = false);

private:
    // Private constructor used by factory methods, view(), and Operation::execute().
    Tensor(std::shared_ptr<Buffer>  buf,
           std::vector<std::size_t> shape,
           std::vector<std::size_t> stride,
           std::size_t              offset,
           DType                    dtype,
           Backend*                 backend);

    // Topological DFS for backward traversal. Defined in tensor.cpp.
    static void topo_dfs(const Tensor&                    t,
                         std::unordered_set<Operation*>&  visited,
                         std::vector<Tensor>&             order);

    std::shared_ptr<Buffer>  buffer_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> stride_;
    std::size_t              offset_        = 0;
    DType                    dtype_         = DType::Float64;
    Backend*                 backend_       = nullptr;  // non-owning; Backend must outlive Tensor

    // Cached at construction from stride_ and shape_. Safe because we have no
    // in-place stride-mutation ops — any layout change creates a new Tensor.
    bool                     is_contiguous_ = true;

    // ── Autograd fields ───────────────────────────────────────────────────────

    bool                               requires_grad_ = false;
    bool                               is_leaf_       = true;

    // Shared across all value-type copies of the same logical tensor. This is the
    // mechanism that makes copy semantics work with gradient accumulation:
    // a grad flowing into saved_inputs_[i] (a copy) updates the user's original.
    // mutable: accumulate_grad() and zero_grad() are logically const operations.
    mutable std::shared_ptr<GradAccumulator> grad_accum_;

    // Non-null for computed tensors; null for leaves and after graph cleanup.
    // Use is_leaf_ to distinguish "true leaf" from "cleaned computed tensor".
    // mutable: backward cleanup nulls it out on const Tensor refs inside saved_inputs_.
    mutable std::shared_ptr<Operation>       grad_op_;

    // Operation::execute() sets autograd fields (is_leaf_, requires_grad_,
    // grad_accum_, grad_op_) on output tensors directly via friend access.
    friend class Operation;
};


// =============================================================================
// ── GradAccumulator — defined after Tensor is complete ───────────────────────
// =============================================================================
//
// Holds the accumulated gradient for a leaf tensor.
// All value-type copies of a logical tensor share one GradAccumulator via
// shared_ptr — the mechanism that makes copy semantics work with autograd.
// Zero runtime overhead when requires_grad is false (grad_accum_ stays null).

struct GradAccumulator {
    Tensor grad_tensor;  // undefined (default-constructed) until first accumulate_grad()
};


// =============================================================================
// ── from_data template body ───────────────────────────────────────────────────
// Defined here because it is a template — all callers must see the body.
// Backend and Buffer are fully defined via the includes above.
// =============================================================================

template<typename T>
Tensor Tensor::from_data(const std::vector<T>&           data,
                          const std::vector<std::size_t>& shape,
                          Backend&                        backend,
                          bool                            requires_grad)
{
    std::size_t n = 1;
    for (auto d : shape) n *= d;
    if (data.size() != n)
        throw std::invalid_argument(
            "Tensor::from_data: data.size() does not match product of shape");

    auto strides = detail::contiguous_strides(shape);
    auto buf = std::make_shared<Buffer>(n * sizeof(T), backend,
                                        static_cast<const void*>(data.data()));
    Tensor t(std::move(buf), shape, std::move(strides), 0,
             dtype_utils::dtype_of<T>::value, &backend);
    if (requires_grad) {
        t.requires_grad_ = true;
        t.grad_accum_    = std::make_shared<GradAccumulator>();
    }
    return t;
}

} // namespace otter
