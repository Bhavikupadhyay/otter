#include "otter/tensor.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <unordered_set>

#include "otter/ops/operation.h"
#include "otter/ops/add_operation.h"
#include "otter/ops/sum_operation.h"

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

// ── Tensor::add / sum (wired here; mul/matmul added in later steps) ───────────

Tensor Tensor::add(const Tensor& other) const {
    return std::make_shared<AddOperation>()->execute({*this, other})[0];
}

Tensor Tensor::sum() const {
    return std::make_shared<SumOperation>()->execute({*this})[0];
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
