#pragma once

#include "otter/ops/operation.h"

namespace otter::ops {

// LogOperation — element-wise natural log: out = log(x).
//
// Backward: d(log(x))/dx = 1/x.
//   grad_x = grad_out / saved_inputs_[0]
// IEEE 754: x=0 → -inf; x<0 → nan — no checks, gradients propagate accordingly.
class LogOperation final : public Operation {
public:
    LogOperation() = default;

    [[nodiscard]] std::vector<Tensor> backward(
        const std::vector<Tensor>& grad_outputs) override;

protected:
    [[nodiscard]] std::vector<Tensor> forward(
        const std::vector<Tensor>& inputs) override;
};

} // namespace otter::ops
