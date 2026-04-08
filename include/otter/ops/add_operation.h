#pragma once

#include "otter/ops/operation.h"

namespace otter::ops {

// AddOperation — element-wise addition: out = a + b.
//
// Supports broadcasting: execute() inserts BroadcastOp nodes when inputs
// have mismatched shapes (via the default NumPy elementwise infer_shapes).
//
// Backward: d(a+b)/da = 1, d(a+b)/db = 1 — gradient flows unchanged.
//   grad_a = grad_out
//   grad_b = grad_out
// (BroadcastOp nodes in the graph handle reduction to original shapes.)
class AddOperation final : public Operation {
public:
    AddOperation() = default;
    [[nodiscard]] bool supports_broadcasting() const noexcept override { return true; }

    [[nodiscard]] std::vector<Tensor> backward(
        const std::vector<Tensor>& grad_outputs) override;

protected:
    [[nodiscard]] std::vector<Tensor> forward(
        const std::vector<Tensor>& inputs) override;
};

} // namespace otter::ops
