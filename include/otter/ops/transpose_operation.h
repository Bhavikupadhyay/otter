#pragma once

#include <cstddef>
#include <vector>

#include "otter/ops/operation.h"

namespace otter::ops {

// TransposeOperation — swaps two dimensions without moving data.
//
// The output shares the same Buffer as the input. Shape and stride entries
// at dim0 and dim1 are swapped — all other metadata is unchanged.
//
// Forward:  swap shape[dim0]↔shape[dim1] and stride[dim0]↔stride[dim1];
//           return view.
// Backward: apply the same swap to the upstream gradient (transpose is
//           self-inverse).
class TransposeOperation final : public Operation {
public:
    TransposeOperation(std::size_t dim0, std::size_t dim1)
        : dim0_(dim0), dim1_(dim1) {}

    [[nodiscard]] std::vector<Tensor> backward(
        const std::vector<Tensor>& grad_outputs) override;

protected:
    [[nodiscard]] std::vector<Tensor> forward(
        const std::vector<Tensor>& inputs) override;

private:
    std::size_t dim0_;
    std::size_t dim1_;
};

} // namespace otter::ops
