#pragma once

#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <vector>

#include "otter/core/dtype.h"
#include "otter/detail/stride_utils.h"
#include "otter/kernel/backend.h"  // full def needed for from_data template body
#include "otter/memory/buffer.h"   // full def needed for from_data template body

namespace otter {

// Tensor — value-type multi-dimensional array.
//
// Multiple Tensors may share one Buffer (view semantics): copies are cheap,
// each Tensor has independent shape/stride/offset metadata.
//
// The Rule of Zero applies: shared_ptr<Buffer> manages the allocation; all
// five special members are compiler-generated and correct.
//
// Autograd fields are added in the Operations step. This is the minimal
// Tensor needed by CPUKernelEngine.
class Tensor {
public:
    // ── Factory ──────────────────────────────────────────────────────────────

    // Allocates a zero-initialised contiguous tensor via backend.
    static Tensor zeros(const std::vector<std::size_t>& shape,
                        Backend& backend,
                        DType dtype = DType::Float64);

    // Copies data into a fresh contiguous tensor.
    // T must have a dtype_utils::dtype_of<T> specialisation (currently double).
    template<typename T>
    static Tensor from_data(const std::vector<T>& data,
                            const std::vector<std::size_t>& shape,
                            Backend& backend);

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
    [[nodiscard]] Backend&    backend() const;  // asserts defined()

    // ── Buffer access (internal — for KernelEngine only) ─────────────────────
    // Raw data is further gated by Passkey<KernelEngine> inside Buffer.
    [[nodiscard]] Buffer&       mutable_buffer();
    [[nodiscard]] const Buffer& buffer() const;

    // ── Layout ───────────────────────────────────────────────────────────────

    // Returns a new Tensor sharing the same Buffer with new shape/stride.
    // No autograd — pure layout alias. The returned Tensor is a fresh value
    // with no grad history.
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
    // Writes value to every element. Asserts is_contiguous (always true for
    // freshly allocated tensors from zeros()). Not differentiable.
    void fill_(double value);

private:
    // Private constructor used by factory methods and view().
    Tensor(std::shared_ptr<Buffer>  buf,
           std::vector<std::size_t> shape,
           std::vector<std::size_t> stride,
           std::size_t              offset,
           DType                    dtype,
           Backend*                 backend);

    std::shared_ptr<Buffer>  buffer_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> stride_;
    std::size_t              offset_        = 0;
    DType                    dtype_         = DType::Float64;
    Backend*                 backend_       = nullptr;  // non-owning; Backend must outlive Tensor
    bool                     is_contiguous_ = true;     // cached at construction
};

// ── from_data template body ────────────────────────────────────────────────────
// Defined here because it is a template — all callers must see the body.
// Backend and Buffer are fully defined via the includes above.

template<typename T>
Tensor Tensor::from_data(const std::vector<T>& data,
                          const std::vector<std::size_t>& shape,
                          Backend& backend)
{
    std::size_t n = 1;
    for (auto d : shape) n *= d;
    assert(data.size() == n && "from_data: data.size() != product of shape");

    auto strides = detail::contiguous_strides(shape);
    auto buf = std::make_shared<Buffer>(n * sizeof(T), backend,
                                         static_cast<const void*>(data.data()));
    return Tensor(std::move(buf), shape, std::move(strides), 0,
                  dtype_utils::dtype_of<T>::value, &backend);
}

} // namespace otter
