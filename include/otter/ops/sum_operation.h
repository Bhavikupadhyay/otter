#pragma once

#include "otter/ops/operation.h"

namespace otter {

// SumOperation — reduce all elements to a scalar: out = sum(a), shape {1}.
//
// Does not support broadcasting (unary reduction — no shape mismatch possible).
//
// Backward: d(sum(x))/dx_i = 1 for all i.
//   grad_input = tensor filled with scalar_grad (fan out to input shape).
// Note: reading the scalar gradient via at({0}) forces a device→host sync on
// GPU backends. A future KernelType::BroadcastScalar kernel avoids this.
class SumOperation final : public Operation {
public:
    SumOperation() = default;
    [[nodiscard]] bool supports_broadcasting() const noexcept override { return false; }

    [[nodiscard]] std::vector<Tensor> backward(
        const std::vector<Tensor>& grad_outputs) override;

protected:
    [[nodiscard]] std::vector<Tensor> forward(
        const std::vector<Tensor>& inputs) override;
};

} // namespace otter
