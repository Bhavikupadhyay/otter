#pragma once

#include "otter/ops/operation.h"

namespace otter::ops {

// NegOperation — element-wise negation: out = -x.
//
// Backward: d(-x)/dx = -1 — gradient is negated.
//   grad_x = -grad_out
// No saved state needed.
class NegOperation final : public Operation {
public:
    NegOperation() = default;

    [[nodiscard]] std::vector<Tensor> backward(
        const std::vector<Tensor>& grad_outputs) override;

protected:
    [[nodiscard]] std::vector<Tensor> forward(
        const std::vector<Tensor>& inputs) override;
};

} // namespace otter::ops
