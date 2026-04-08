#pragma once

#include "otter/ops/operation.h"

namespace otter::ops {

// DivOperation — element-wise division: out = a / b.
//
// Supports broadcasting via the default NumPy elementwise infer_shapes.
//
// Backward: product-rule quotient derivative.
//   grad_a = grad_out / b
//   grad_b = -grad_out * a / (b * b)
// IEEE 754: b=0 produces ±inf or nan in both forward and backward — no checks.
// (BroadcastOp nodes in the graph handle reduction to original shapes.)
class DivOperation final : public Operation {
public:
    DivOperation() = default;
    [[nodiscard]] bool supports_broadcasting() const noexcept override { return true; }

    [[nodiscard]] std::vector<Tensor> backward(
        const std::vector<Tensor>& grad_outputs) override;

protected:
    [[nodiscard]] std::vector<Tensor> forward(
        const std::vector<Tensor>& inputs) override;
};

} // namespace otter::ops
