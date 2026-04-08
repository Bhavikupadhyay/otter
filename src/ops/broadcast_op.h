#pragma once

// Internal header — not part of the public include/otter/ surface.
// Included only by src/ops/operation.cpp and src/ops/broadcast_op.cpp.

#include "otter/ops/operation.h"

namespace otter {

// BroadcastOp — view operation that expands a tensor to a larger shape using
// stride-zero semantics. Inserted automatically by execute() when inputs to
// a supports_broadcasting() op have mismatched shapes.
//
// Forward:  creates a stride-zero view — no data copy, O(1).
//           The view shares the original buffer; repeating dims read the same
//           element via stride 0.
//
// Backward: reduces the incoming gradient over the broadcast axes back to the
//           original shape by summing (dispatch_unary KernelType::ReduceTo).
//           dispatch_unary(ReduceTo, grad, grad_input) accumulates grad (at
//           broadcast shape) into grad_input (at original shape).
//
// supports_broadcasting() returns false: BroadcastOp itself never triggers
// further broadcasting, preventing infinite recursion in execute().
class BroadcastOp final : public Operation {
public:
    explicit BroadcastOp(std::vector<std::size_t> target_shape)
        : target_shape_(std::move(target_shape)) {}

    [[nodiscard]] bool supports_broadcasting() const noexcept override { return false; }

    [[nodiscard]] std::vector<Tensor> backward(
        const std::vector<Tensor>& grad_outputs) override;

protected:
    [[nodiscard]] std::vector<Tensor> forward(
        const std::vector<Tensor>& inputs) override;

private:
    std::vector<std::size_t> target_shape_;
};

} // namespace otter
