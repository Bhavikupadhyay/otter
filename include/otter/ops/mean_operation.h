#pragma once

#include "otter/ops/operation.h"

namespace otter::ops {

// MeanOperation — reduce all elements to scalar mean: out = sum(x) / numel(x).
// Output shape: {1}.
//
// Backward: d(mean(x))/dx_i = 1/n for all i.
//   grad_input is a tensor of the same shape as x, filled with grad_out[0] / n.
//   Same device→host sync note as SumOperation: at({0}) in backward forces
//   a host read of the gradient scalar. Acceptable for CPU; revisit for CUDA.
class MeanOperation final : public Operation {
public:
    MeanOperation() = default;
    [[nodiscard]] bool supports_broadcasting() const noexcept override { return false; }

    [[nodiscard]] std::vector<Tensor> backward(
        const std::vector<Tensor>& grad_outputs) override;

protected:
    [[nodiscard]] std::vector<Tensor> forward(
        const std::vector<Tensor>& inputs) override;
};

} // namespace otter::ops
