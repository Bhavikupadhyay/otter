#include "otter/tensor.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "otter/ops/operation.h"
#include "ops/broadcast_op.h"
#include "otter/ops/add_operation.h"
#include "otter/ops/div_operation.h"
#include "otter/ops/exp_operation.h"
#include "otter/ops/log_operation.h"
#include "otter/ops/matmul_operation.h"
#include "otter/ops/mul_operation.h"
#include "otter/ops/neg_operation.h"
#include "otter/ops/relu_operation.h"
#include "otter/ops/reshape_operation.h"
#include "otter/ops/slice_operation.h"
#include "otter/ops/sqrt_operation.h"
#include "otter/ops/sub_operation.h"
#include "otter/ops/sum_operation.h"
#include "otter/ops/transpose_operation.h"

namespace otter {

// ── Private constructor ───────────────────────────────────────────────────────

Tensor::Tensor(std::shared_ptr<Buffer>  buf,
               std::vector<std::size_t> shape,
               std::vector<std::size_t> stride,
               std::size_t              offset,
               DType                    dtype,
               Backend*                 backend)
    : buffer_(std::move(buf))
    , shape_(std::move(shape))
    , stride_(std::move(stride))
    , offset_(offset)
    , dtype_(dtype)
    , backend_(backend)
    , is_contiguous_(stride_ == detail::contiguous_strides(shape_))
{}

// ── Factory ───────────────────────────────────────────────────────────────────

Tensor Tensor::zeros(const std::vector<std::size_t>& shape,
                     Backend& backend,
                     DType dtype,
                     bool  requires_grad)
{
    std::size_t nbytes = dtype_utils::size_of(dtype);
    for (auto d : shape) nbytes *= d;

    auto strides = detail::contiguous_strides(shape);
    auto buf     = std::make_shared<Buffer>(nbytes, backend);  // zero-initialised by Buffer ctor
    Tensor t(std::move(buf), shape, std::move(strides), 0, dtype, &backend);
    if (requires_grad) {
        t.requires_grad_ = true;
        t.grad_accum_    = std::make_shared<GradAccumulator>();
    }
    return t;
}

Tensor Tensor::ones(const std::vector<std::size_t>& shape,
                    Backend& backend,
                    DType dtype,
                    bool  requires_grad)
{
    Tensor t = zeros(shape, backend, dtype, requires_grad);
    t.fill_(1.0);
    return t;
}

Tensor Tensor::full(const std::vector<std::size_t>& shape,
                    double value,
                    Backend& backend,
                    DType dtype,
                    bool  requires_grad)
{
    Tensor t = zeros(shape, backend, dtype, requires_grad);
    t.fill_(value);
    return t;
}

Tensor Tensor::zeros_like(const Tensor& t, bool requires_grad) {
    return zeros(t.shape(), t.backend(), t.dtype(), requires_grad);
}

Tensor Tensor::ones_like(const Tensor& t, bool requires_grad) {
    return ones(t.shape(), t.backend(), t.dtype(), requires_grad);
}

// ── Accessors ─────────────────────────────────────────────────────────────────

std::size_t Tensor::numel() const noexcept {
    std::size_t n = 1;
    for (auto d : shape_) n *= d;
    return n;
}

Backend& Tensor::backend() const {
    if (!defined())
        throw std::runtime_error("Tensor::backend() called on undefined tensor");
    return *backend_;
}

Buffer& Tensor::mutable_buffer() {
    assert(defined() && "Tensor::mutable_buffer() called on undefined tensor");
    return *buffer_;
}

const Buffer& Tensor::buffer() const {
    assert(defined() && "Tensor::buffer() called on undefined tensor");
    return *buffer_;
}

// ── Layout ────────────────────────────────────────────────────────────────────

Tensor Tensor::view(std::vector<std::size_t> new_shape,
                    std::vector<std::size_t> new_stride) const
{
    if (!defined())
        throw std::runtime_error("Tensor::view() called on undefined tensor");
    assert(new_shape.size() == new_stride.size() &&
           "Tensor::view(): shape and stride must have the same number of dimensions");
    return Tensor(buffer_, std::move(new_shape), std::move(new_stride),
                  offset_, dtype_, backend_);
}

Tensor Tensor::view(std::vector<std::size_t> new_shape,
                    std::vector<std::size_t> new_stride,
                    std::size_t              new_offset) const
{
    if (!defined())
        throw std::runtime_error("Tensor::view() called on undefined tensor");
    assert(new_shape.size() == new_stride.size() &&
           "Tensor::view(): shape and stride must have the same number of dimensions");
    return Tensor(buffer_, std::move(new_shape), std::move(new_stride),
                  new_offset, dtype_, backend_);
}

Tensor Tensor::contiguous() const {
    if (!defined())
        throw std::runtime_error("Tensor::contiguous() called on undefined tensor");
    if (is_contiguous_) return *this;
    // Allocate fresh contiguous buffer and copy elements via the strided copy kernel.
    // The returned tensor is a plain value — no grad history.
    Tensor out = zeros(shape_, *backend_, dtype_);
    backend_->kernel_engine()->dispatch_copy(*this, out);
    return out;
}

// ── Element access ────────────────────────────────────────────────────────────

double Tensor::at(std::initializer_list<std::size_t> indices) const {
    if (!defined())
        throw std::runtime_error("Tensor::at() called on undefined tensor");
    if (indices.size() != shape_.size())
        throw std::out_of_range(
            "Tensor::at(): wrong number of indices (got " +
            std::to_string(indices.size()) + ", tensor has " +
            std::to_string(shape_.size()) + " dimensions)");

    std::size_t flat = offset_;
    std::size_t dim  = 0;
    for (std::size_t idx : indices) {
        if (idx >= shape_[dim])
            throw std::out_of_range(
                "Tensor::at(): index " + std::to_string(idx) +
                " out of bounds for dimension " + std::to_string(dim) +
                " with size " + std::to_string(shape_[dim]));
        flat += idx * stride_[dim++];
    }
    // Device-explicit element read — on CUDA this is an explicit device→host transfer.
    return backend_->kernel_engine()->dispatch_element_read(*this, flat);
}

// ── In-place fill ─────────────────────────────────────────────────────────────

void Tensor::fill_(double value) {
    if (!defined())
        throw std::runtime_error("Tensor::fill_() called on undefined tensor");
    assert(is_contiguous_ && "Tensor::fill_() requires a contiguous tensor");
    assert(buffer_.use_count() == 1 &&
           "Tensor::fill_() requires unique buffer ownership — "
           "do not call fill_() on views or copies sharing the same buffer");
    backend_->kernel_engine()->dispatch_fill(*this, value);
}

// ── Autograd methods ──────────────────────────────────────────────────────────

Tensor Tensor::grad() const noexcept {
    if (!grad_accum_) return Tensor{};
    return grad_accum_->grad_tensor;
}

Tensor Tensor::detach() const noexcept {
    Tensor t         = *this;   // shallow copy: shares Buffer, shape, stride, dtype, backend
    t.requires_grad_ = false;
    t.grad_accum_    = nullptr;
    t.grad_op_       = nullptr;
    // is_leaf_ intentionally NOT changed: detach does not make a computed tensor a leaf.
    // is_leaf_ is strictly "did the user create this tensor directly via zeros/from_data".
    return t;
}

void Tensor::zero_grad() noexcept {
    if (grad_accum_) grad_accum_->grad_tensor = Tensor{};
}

void Tensor::accumulate_grad(const Tensor& incoming) const {
    assert(requires_grad_ && "accumulate_grad called on tensor that does not require grad");
    if (!grad_accum_) grad_accum_ = std::make_shared<GradAccumulator>();
    if (!grad_accum_->grad_tensor.defined()) {
        grad_accum_->grad_tensor = Tensor::zeros(shape_, *backend_, dtype_);
    }
    assert(incoming.shape() == shape_ &&
           "accumulate_grad: incoming gradient shape does not match tensor shape");
    // detach() prevents a second-order grad graph through the accumulation add().
    // Without it, add() would wire a new AddOperation whose saved_inputs_ holds
    // references back into the grad chain, creating reference cycles.
    grad_accum_->grad_tensor = grad_accum_->grad_tensor.add(incoming.detach());
}

// ── Tensor operations ─────────────────────────────────────────────────────────

Tensor Tensor::add(const Tensor& other) const {
    return std::make_shared<AddOperation>()->execute({*this, other})[0];
}

Tensor Tensor::sub(const Tensor& other) const {
    return std::make_shared<SubOperation>()->execute({*this, other})[0];
}

Tensor Tensor::mul(const Tensor& other) const {
    return std::make_shared<MulOperation>()->execute({*this, other})[0];
}

Tensor Tensor::div(const Tensor& other) const {
    return std::make_shared<DivOperation>()->execute({*this, other})[0];
}

Tensor Tensor::neg() const {
    return std::make_shared<NegOperation>()->execute({*this})[0];
}

Tensor Tensor::exp() const {
    return std::make_shared<ExpOperation>()->execute({*this})[0];
}

Tensor Tensor::log() const {
    return std::make_shared<LogOperation>()->execute({*this})[0];
}

Tensor Tensor::sqrt() const {
    return std::make_shared<SqrtOperation>()->execute({*this})[0];
}

Tensor Tensor::relu() const {
    return std::make_shared<ReluOperation>()->execute({*this})[0];
}

Tensor Tensor::matmul(const Tensor& other) const {
    return std::make_shared<MatMulOperation>()->execute({*this, other})[0];
}

Tensor Tensor::reshape(std::vector<std::size_t> new_shape) const {
    return std::make_shared<ReshapeOperation>(std::move(new_shape))->execute({*this})[0];
}

Tensor Tensor::transpose(std::size_t dim0, std::size_t dim1) const {
    return std::make_shared<TransposeOperation>(dim0, dim1)->execute({*this})[0];
}

Tensor Tensor::slice(std::size_t dim, std::size_t start, std::size_t length) const {
    return std::make_shared<SliceOperation>(dim, start, length)->execute({*this})[0];
}

Tensor Tensor::broadcast_to(std::vector<std::size_t> target_shape) const {
    return std::make_shared<BroadcastOp>(std::move(target_shape))->execute({*this})[0];
}

Tensor Tensor::sum() const {
    return std::make_shared<SumOperation>()->execute({*this})[0];
}

// ── Debug output ──────────────────────────────────────────────────────────────

void Tensor::print(const std::string& label) const {
    if (!label.empty())
        std::cout << label << "\n";

    if (!defined()) {
        std::cout << "<undefined tensor>\n";
        return;
    }

    // Shape
    std::cout << "shape: [";
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << shape_[i];
    }
    std::cout << "]";

    // DType
    std::cout << "  dtype: ";
    switch (dtype_) {
        case DType::Float64: std::cout << "Float64"; break;
        default:             std::cout << "unknown"; break;
    }
    std::cout << "\n";

    // Values — row-major iteration via the same offset+stride logic as to_vector.
    const std::size_t n    = numel();
    const std::size_t ndim = shape_.size();
    std::vector<std::size_t> coords(ndim, 0);
    for (std::size_t i = 0; i < n; ++i) {
        // Print opening brackets for each new outermost run
        if (ndim > 1) {
            for (std::size_t d = 0; d < ndim - 1; ++d) {
                if (coords[d] == 0) std::cout << "[";
            }
        }

        std::size_t flat = offset_;
        for (std::size_t d = 0; d < ndim; ++d) flat += coords[d] * stride_[d];
        std::cout << backend_->kernel_engine()->dispatch_element_read(*this, flat);

        // Increment odometer and print closing brackets / separators
        bool carried = false;
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            const auto ud = static_cast<std::size_t>(d);
            if (++coords[ud] < shape_[ud]) break;
            coords[ud] = 0;
            carried = true;
            if (d > 0) std::cout << "]";
        }
        if (i < n - 1) {
            if (carried && ndim > 1) std::cout << ",\n ";
            else                     std::cout << ", ";
        }
    }
    if (ndim > 1) std::cout << "]";
    std::cout << "\n";
}

// ── Topological DFS (static — accesses private autograd fields) ───────────────

void Tensor::topo_dfs(const Tensor&                   t,
                      std::unordered_set<Operation*>& visited,
                      std::vector<Tensor>&             order)
{
    // Stop at leaves (user-created tensors) and at nodes whose grad_op_ was
    // cleared by a previous backward pass. is_leaf_ distinguishes the two states:
    // "true leaf" vs "computed tensor after cleanup" (both have grad_op_==nullptr).
    if (t.is_leaf_ || !t.grad_op_) return;
    Operation* op = t.grad_op_.get();
    if (visited.count(op)) return;
    visited.insert(op);
    for (const Tensor& inp : t.grad_op_->inputs())
        topo_dfs(inp, visited, order);
    order.push_back(t);  // post-order: children before parents
}

// ── Tensor::backward ──────────────────────────────────────────────────────────

void Tensor::backward(bool retain_graph) {
    if (!defined())
        throw std::runtime_error("backward: called on undefined Tensor");
    if (is_leaf_)
        throw std::runtime_error(
            "backward: called on a leaf Tensor — call backward on a computed result");
    if (!grad_op_)
        throw std::runtime_error(
            "backward: grad_op_ is null — graph was already cleared. "
            "Use retain_graph=true to call backward twice.");
    // Seed: ones of the same shape (correct for scalar loss and any shape).
    Tensor seed = Tensor::zeros(shape_, *backend_, dtype_);
    seed.fill_(1.0);
    backward(std::move(seed), retain_graph);
}

void Tensor::backward(Tensor seed, bool retain_graph) {
    if (!defined())
        throw std::runtime_error("backward: called on undefined Tensor");

    // ── 1. Build topological order (DFS post-order → reverse = loss first) ───
    std::vector<Tensor>            order;
    std::unordered_set<Operation*> visited;
    topo_dfs(*this, visited, order);
    std::reverse(order.begin(), order.end());

    // ── 2. Clear intermediate (non-leaf) gradients ────────────────────────────
    // On a second backward call (retain_graph=true first), intermediate nodes
    // still hold gradients from the previous pass. Clear them so each backward
    // starts fresh for intermediates. Leaf gradients are NOT cleared here —
    // they accumulate across multiple backward calls (the expected behaviour).
    for (const Tensor& node : order) {
        if (!node.is_leaf_ && node.grad_accum_)
            node.grad_accum_->grad_tensor = Tensor{};
    }

    // ── 3. Seed this tensor's gradient ───────────────────────────────────────
    // Assign (not accumulate) the seed. After clearing intermediates above,
    // this tensor's grad_accum_ is empty; assigning directly is equivalent
    // to a fresh start for this node's gradient.
    if (!grad_accum_) grad_accum_ = std::make_shared<GradAccumulator>();
    grad_accum_->grad_tensor = std::move(seed);

    // ── 4. Traverse in reverse topo order ─────────────────────────────────────
    for (const Tensor& node : order) {
        if (!node.grad_op_) continue;
        if (!node.grad_accum_ || !node.grad_accum_->grad_tensor.defined()) continue;

        const Tensor& node_grad = node.grad_accum_->grad_tensor;

        // Run the operation's backward pass.
        auto input_grads = node.grad_op_->backward({node_grad});

        // Accumulate each input gradient into the corresponding saved input.
        const auto& saved = node.grad_op_->inputs();
        for (std::size_t i = 0; i < saved.size() && i < input_grads.size(); ++i) {
            if (saved[i].requires_grad() && input_grads[i].defined())
                saved[i].accumulate_grad(input_grads[i]);
        }

        // ── 4. Graph cleanup (unless retain_graph) ────────────────────────────
        if (!retain_graph) node.grad_op_->clear_saved();
    }

    // Null out this node's grad_op_ after cleanup so a second backward() throws
    // rather than silently double-counting gradients.
    if (!retain_graph) grad_op_ = nullptr;
}

} // namespace otter
