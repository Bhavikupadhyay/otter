#pragma once

#include <cstddef>
#include <vector>

#include "otter/ops/operation.h"

namespace otter::ops {

// SliceOperation — extracts a contiguous sub-range along one dimension.
//
// The output shares the same Buffer as the input. Only offset and shape[dim]
// change; strides are unchanged.
//
// Forward:  new_offset = offset + start * stride[dim], new shape[dim] = length.
// Backward: scatter the upstream gradient back into a zeros tensor at the
//           slice position using strided copy. Gradient is zero outside the
//           slice region (correct for all non-slice elements).
class SliceOperation final : public Operation {
public:
    SliceOperation(std::size_t dim, std::size_t start, std::size_t length)
        : dim_(dim), start_(start), length_(length) {}

    [[nodiscard]] std::vector<Tensor> backward(
        const std::vector<Tensor>& grad_outputs) override;

protected:
    [[nodiscard]] std::vector<Tensor> forward(
        const std::vector<Tensor>& inputs) override;

private:
    std::size_t dim_;
    std::size_t start_;
    std::size_t length_;
};

} // namespace otter::ops
