#pragma once

#include "otter/ops/operation.h"

namespace otter {

// SqrtOperation — element-wise square root: out = sqrt(x).
//
// Backward: d(sqrt(x))/dx = 1 / (2 * sqrt(x)).
//   grad_x = grad_out / (2 * saved_outputs_[0])   (reuse forward output)
// IEEE 754: x<0 → nan; x=0 → grad = +inf (0 / (2*0)).
class SqrtOperation final : public Operation {
public:
    SqrtOperation() = default;

    [[nodiscard]] std::vector<Tensor> backward(
        const std::vector<Tensor>& grad_outputs) override;

protected:
    [[nodiscard]] std::vector<Tensor> forward(
        const std::vector<Tensor>& inputs) override;
};

} // namespace otter
