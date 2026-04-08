#pragma once

#include <cstddef>
#include <vector>

#include "otter/ops/operation.h"

namespace otter::ops {

// ReshapeOperation — changes shape without moving data.
//
// Precondition: input must be contiguous. If it is not, call contiguous()
// first. The output shares the same Buffer — no copy is performed.
//
// Forward:  validate numel preserved + contiguous; return view with new
//           contiguous strides.
// Backward: reshape the upstream gradient back to the original shape.
//           The grad is contiguous by construction, so a raw view suffices.
class ReshapeOperation final : public Operation {
public:
    explicit ReshapeOperation(std::vector<std::size_t> target_shape)
        : target_shape_(std::move(target_shape)) {}

    [[nodiscard]] std::vector<Tensor> backward(
        const std::vector<Tensor>& grad_outputs) override;

protected:
    [[nodiscard]] std::vector<Tensor> forward(
        const std::vector<Tensor>& inputs) override;

private:
    std::vector<std::size_t> target_shape_;
};

} // namespace otter::ops
