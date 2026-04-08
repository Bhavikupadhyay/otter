#include "otter/ops/operation.h"
#include "broadcast_op.h"

#include <cassert>
#include <stdexcept>

#include "otter/autograd/no_grad_guard.h"
#include "otter/detail/broadcast.h"

namespace otter {

// ── BroadcastSpec::infer_shapes default — NumPy elementwise rules ─────────────
//
// All inputs broadcast to the same output shape.
// Compute pairwise broadcast across all inputs; assign that shape to every input.
// Called only when supports_broadcasting() == true.

BroadcastSpec Operation::infer_shapes(const std::vector<Tensor>& inputs) const {
    assert(inputs.size() >= 2 && "infer_shapes: called with fewer than 2 inputs");
    std::vector<std::size_t> out_shape = inputs[0].shape();
    for (std::size_t i = 1; i < inputs.size(); ++i)
        out_shape = detail::broadcast_shape(out_shape, inputs[i].shape());
    BroadcastSpec spec;
    spec.input_target_shapes.assign(inputs.size(), out_shape);
    return spec;
}


// ── Operation::validate_binary ────────────────────────────────────────────────

void Operation::validate_binary(const Tensor& a, const Tensor& b, const char* op_name) {
    if (!a.defined() || !b.defined())
        throw std::runtime_error(std::string(op_name) + ": input Tensor is undefined");
    if (a.dtype() != b.dtype())
        throw std::runtime_error(std::string(op_name) + ": inputs must have the same dtype");
    if (a.shape() != b.shape())
        throw std::runtime_error(std::string(op_name) + ": inputs must have the same shape");
}


// ── Operation::execute ────────────────────────────────────────────────────────
//
// This is the sole owner of graph wiring. forward() and backward() are clean.
//
// Steps:
//   1. If supports_broadcasting(): call infer_shapes(), insert BroadcastOp nodes
//      for any input whose shape doesn't match the target.
//   2. Save (broadcast-adjusted) inputs to saved_inputs_.
//   3. Call forward() on the adjusted inputs.
//   4. If any adjusted input has requires_grad AND grad tracking is enabled:
//      set is_leaf_=false, requires_grad_=true, allocate grad_accum_, set grad_op_.

std::vector<Tensor> Operation::execute(const std::vector<Tensor>& inputs) {
    // ── 1. Broadcast insertion ─────────────────────────────────────────────────
    std::vector<Tensor> actual_inputs = inputs;

    if (supports_broadcasting() && inputs.size() >= 2) {
        BroadcastSpec spec = infer_shapes(inputs);

        bool needs_broadcast = false;
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            if (inputs[i].shape() != spec.input_target_shapes[i]) {
                needs_broadcast = true;
                break;
            }
        }

        if (needs_broadcast) {
            actual_inputs.clear();
            actual_inputs.reserve(inputs.size());
            for (std::size_t i = 0; i < inputs.size(); ++i) {
                if (inputs[i].shape() != spec.input_target_shapes[i]) {
                    // BroadcastOp::execute() wires its own grad_op_ so backward
                    // flows through it automatically — no special casing needed.
                    actual_inputs.push_back(
                        std::make_shared<BroadcastOp>(spec.input_target_shapes[i])
                            ->execute({inputs[i]})[0]);
                } else {
                    actual_inputs.push_back(inputs[i]);
                }
            }
        }
    }

    // ── 2. Save inputs + run forward ──────────────────────────────────────────
    // saved_inputs_ holds broadcast-adjusted inputs, NOT the originals.
    // MulOp backward needs b at broadcast shape (same shape as grad_out).
    saved_inputs_  = actual_inputs;
    auto outputs   = forward(actual_inputs);
    // saved_outputs_ holds the plain forward outputs (before grad_op_ wiring).
    // Used by ops whose backward reuses the output value (exp, sqrt).
    saved_outputs_ = outputs;

    // ── 3. Decide whether to wire the grad graph ───────────────────────────────
    // Skip entirely if NoGradGuard is active — outputs are plain tensors.
    bool any_grad = false;
    if (NoGradGuard::grad_mode()) {
        for (const auto& inp : actual_inputs) {
            if (inp.requires_grad()) { any_grad = true; break; }
        }
    }

    // ── 4. Wire grad graph on outputs ─────────────────────────────────────────
    if (any_grad) {
        auto self = shared_from_this();
        for (auto& out : outputs) {
            out.is_leaf_       = false;
            out.requires_grad_ = true;
            out.grad_accum_    = std::make_shared<GradAccumulator>();
            out.grad_op_       = self;
        }
    }
    return outputs;
}

} // namespace otter
