#pragma once

#include "otter/ops/operation.h"

namespace otter::ops {

// MatMulOperation — batched matrix multiply: out = A @ B.
//
// Shape rules (NumPy / PyTorch convention):
//   A: [..., M, K]   B: [..., K, N]   out: [..., M, N]
//   Batch dims (...) are broadcast against each other.
//   The inner K dimension must match exactly.
//
// Supports batch broadcasting: execute() inserts BroadcastOp nodes for any
// batch dim that needs expanding (via the overridden infer_shapes below).
//
// Backward (grad has shape [..., M, N]):
//   grad_A = grad @ B^T  →  shape [..., M, K]
//   grad_B = A^T @ grad  →  shape [..., K, N]
// Both multiplications run under NoGradGuard.
// Transpose is achieved by a stride-swapping view of the last two dims.
class MatMulOperation final : public Operation {
public:
    MatMulOperation() = default;

    [[nodiscard]] bool supports_broadcasting() const noexcept override { return true; }

    // Override: broadcast batch dims only; validate K; return per-input target shapes.
    [[nodiscard]] BroadcastSpec infer_shapes(
        const std::vector<Tensor>& inputs) const override;

    [[nodiscard]] std::vector<Tensor> backward(
        const std::vector<Tensor>& grad_outputs) override;

protected:
    [[nodiscard]] std::vector<Tensor> forward(
        const std::vector<Tensor>& inputs) override;
};

} // namespace otter::ops
