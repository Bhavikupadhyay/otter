#pragma once

#include <cstddef>
#include <memory>
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

    // Drop saved input and output references — releases Buffer refcounts.
    // Called by Tensor::backward() after the gradient has been propagated (retain_graph=false).
    void clear_saved() noexcept { saved_inputs_.clear(); saved_outputs_.clear(); }

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
};

} // namespace otter::ops
