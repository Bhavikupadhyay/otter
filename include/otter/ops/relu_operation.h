#pragma once

#include "otter/ops/operation.h"

namespace otter::ops {

// ReluOperation — rectified linear unit: out = max(0, x).
//
// Backward: grad_x = grad_out * (x > 0 ? 1.0 : 0.0).
//   At x=0: gradient is 0.0 (right-hand derivative / PyTorch convention).
//   Uses saved_inputs_[0] via the ReluMask kernel to produce the derivative mask.
class ReluOperation final : public Operation {
public:
    ReluOperation() = default;

    [[nodiscard]] std::vector<Tensor> backward(
        const std::vector<Tensor>& grad_outputs) override;

protected:
    [[nodiscard]] std::vector<Tensor> forward(
        const std::vector<Tensor>& inputs) override;
};

} // namespace otter::ops
