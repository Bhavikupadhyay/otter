#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "otter/tensor.h"  // full Tensor definition — saved_inputs_, backward args

namespace otter::ops {

// BroadcastSpec — returned by Operation::infer_shapes().
// input_target_shapes[i] is the shape that input i must have before forward() runs.
// execute() compares each input's actual shape against this target and inserts a
// BroadcastOp node for any input that needs expanding.
struct BroadcastSpec {
    std::vector<std::vector<std::size_t>> input_target_shapes;
};


// Operation — abstract base for all differentiable operations.
//
// Public contract:
//   execute(inputs)  — called by Tensor::add/mul/sum/matmul.
//                      Handles broadcast insertion, forward(), and graph wiring.
//                      Never overridden — it is the sole owner of graph state.
//   backward(grads)  — pure virtual. Implemented by each concrete op.
//                      Must not do any graph wiring.
//
// Subclass contract:
//   forward(inputs)  — pure virtual, protected. Compute the output tensor(s).
//                      Must not do any graph wiring.
//   supports_broadcasting() — override to true for binary elementwise ops.
//   infer_shapes(inputs)    — override for ops with non-elementwise shape rules
//                             (e.g. MatMulOperation).
//
// execute() owns ALL autograd field mutations (is_leaf_, requires_grad_,
// grad_accum_, grad_op_) on output tensors. forward() and backward() are clean.
//
// enable_shared_from_this: execute() calls shared_from_this() to store a
// stable shared_ptr<Operation> in each output's grad_op_. Operations must
// always be heap-allocated via make_shared.
class Operation : public std::enable_shared_from_this<Operation> {
public:
    virtual ~Operation() = default;

    Operation(const Operation&)            = delete;
    Operation& operator=(const Operation&) = delete;
    Operation(Operation&&)                 = delete;
    Operation& operator=(Operation&&)      = delete;

    // ── Public entry point ────────────────────────────────────────────────────
    //
    // Runs broadcast insertion → forward() → graph wiring.
    // Returns one Tensor per output (usually one; multi-output reserved for future).
    [[nodiscard]] std::vector<Tensor> execute(const std::vector<Tensor>& inputs);

    // Read access to saved inputs — used by Tensor::backward() traversal.
    [[nodiscard]] const std::vector<Tensor>& inputs()  const noexcept { return saved_inputs_; }

    // Read access to saved outputs — used by backward() implementations that need
    // to reuse the forward output (e.g. ExpOperation, SqrtOperation).
    [[nodiscard]] const std::vector<Tensor>& outputs() const noexcept { return saved_outputs_; }

    // Thread-safe snapshot of saved inputs — returns a by-value copy under saved_mtx_.
    //
    // Used by Tensor::backward() Phase 4 instead of inputs() (which returns const&).
    // If clear_saved() runs on another thread between when we snapshot and when we call
    // op->backward(), the snapshot copy is still valid; our backward() call simply
    // uses the saved inputs captured before cleanup.
    //
    // inputs() (returning const&) is kept for topo_dfs(), which runs in Phase 1 before
    // any Phase 5 cleanup — no race exists there and the by-value overhead is unnecessary.
    [[nodiscard]] std::vector<Tensor> snapshot_inputs() const {
        std::lock_guard<std::mutex> lock(saved_mtx_);
        return saved_inputs_;
    }

    // Drop saved input and output references — releases Buffer refcounts.
    // Called by Tensor::backward() after all gradients are accumulated (retain_graph=false).
    //
    // Before clearing saved_inputs_, we explicitly reset grad_op_ on each saved Tensor
    // copy. This breaks the reference chain:
    //   Operation → saved_inputs_[i] (Tensor copy) → grad_op_ → (another Operation)
    // Without this reset, each Tensor copy in saved_inputs_ would hold a shared_ptr to
    // the next Operation in the graph, keeping the entire chain alive until all
    // saved_inputs_ vectors destruct. Explicit reset frees Operations as soon as they
    // are no longer reachable from any Tensor the caller holds.
    //
    // Acquires saved_mtx_ to protect against concurrent snapshot_inputs() calls on
    // another thread (e.g. two backward passes sharing an intermediate Operation).
    //
    // ops::Operation is a friend of Tensor (tensor.h), so t.grad_op_.reset() is valid.
    void clear_saved() noexcept {
        std::lock_guard<std::mutex> lock(saved_mtx_);
        for (auto& t : saved_inputs_) t.grad_op_.reset();
        saved_inputs_.clear();
        saved_outputs_.clear();
    }

    // ── Backward pass — implemented by each concrete op ───────────────────────
    //
    // grad_outputs[i] is the incoming gradient for output i.
    // Returns one gradient Tensor per saved input.
    // Must not wire any grad graph (all results should be produced without
    // requires_grad, or via NoGradGuard, or should be detach()'d before return).
    [[nodiscard]] virtual std::vector<Tensor> backward(
        const std::vector<Tensor>& grad_outputs) = 0;

    // ── Broadcasting interface ────────────────────────────────────────────────
    //
    // supports_broadcasting(): if true, execute() runs infer_shapes() and inserts
    // BroadcastOp nodes before calling forward(). Default: false.
    //
    // infer_shapes(): given actual inputs, returns the shape each input must have
    // for the kernel. Default (NumPy elementwise): all inputs expand to the same
    // output shape. Override for ops with different rules (e.g. matmul).
    [[nodiscard]] virtual bool supports_broadcasting() const noexcept { return false; }
    [[nodiscard]] virtual BroadcastSpec infer_shapes(const std::vector<Tensor>& inputs) const;

protected:
    Operation() = default;

    // Compute the output tensor(s). Receives broadcast-adjusted inputs.
    [[nodiscard]] virtual std::vector<Tensor> forward(
        const std::vector<Tensor>& inputs) = 0;

    // Shared input validation for binary ops.
    static void validate_binary(const Tensor& a, const Tensor& b, const char* op_name);

    // Saved inputs from execute() — accessed by backward() and topo_dfs().
    // Contents are the broadcast-adjusted inputs (same shape as grad_outputs).
    // mutable so clear_saved() can be called on a const Operation ref.
    mutable std::vector<Tensor> saved_inputs_;

    // Saved outputs from execute() — used by ops whose backward needs the forward
    // output value (exp: grad * out; sqrt: grad / (2 * out)).
    // These are the plain output tensors before grad_op_ wiring.
    mutable std::vector<Tensor> saved_outputs_;

    // Protects saved_inputs_ and saved_outputs_ against concurrent snapshot_inputs()
    // and clear_saved() calls (e.g. two backward passes traversing a shared Operation).
    // mutable: clear_saved() is called on const Operation refs from the backward traversal.
    mutable std::mutex saved_mtx_;
};

} // namespace otter::ops
