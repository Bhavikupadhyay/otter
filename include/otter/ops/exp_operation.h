#pragma once

#include "otter/ops/operation.h"

namespace otter {

// ExpOperation — element-wise exponential: out = exp(x).
//
// Backward: d(exp(x))/dx = exp(x).
//   grad_x = grad_out * saved_outputs_[0]   (reuse forward output, no recompute)
// IEEE 754: x > 709.78 → +inf; x < -745 → 0.0.
class ExpOperation final : public Operation {
public:
    ExpOperation() = default;

    [[nodiscard]] std::vector<Tensor> backward(
        const std::vector<Tensor>& grad_outputs) override;

protected:
    [[nodiscard]] std::vector<Tensor> forward(
        const std::vector<Tensor>& inputs) override;
};

} // namespace otter
