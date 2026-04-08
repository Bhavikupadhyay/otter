#pragma once

#include "otter/ops/operation.h"

namespace otter {

// MulOperation — element-wise multiplication: out = a * b.
//
// Supports broadcasting: execute() inserts BroadcastOp nodes when inputs
// have mismatched shapes (via the default NumPy elementwise infer_shapes).
//
// Backward (product rule):
//   grad_a = grad_out * b
//   grad_b = grad_out * a
// Computed under NoGradGuard so the backward multiplications do not
// build a second-order gradient graph.
// (BroadcastOp nodes in the graph handle reduction to original shapes.)
class MulOperation final : public Operation {
public:
    MulOperation() = default;
    [[nodiscard]] bool supports_broadcasting() const noexcept override { return true; }

    [[nodiscard]] std::vector<Tensor> backward(
        const std::vector<Tensor>& grad_outputs) override;

protected:
    [[nodiscard]] std::vector<Tensor> forward(
        const std::vector<Tensor>& inputs) override;
};

} // namespace otter
