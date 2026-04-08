#include "otter/ops/transpose_operation.h"

#include <cassert>
#include <stdexcept>

namespace otter {

std::vector<Tensor> TransposeOperation::forward(const std::vector<Tensor>& inputs) {
    assert(inputs.size() == 1 && "TransposeOperation: requires exactly 1 input");
    const Tensor& in = inputs[0];

    if (!in.defined())
        throw std::runtime_error("TransposeOperation: input is undefined");

    const std::size_t ndim = in.shape().size();
    if (ndim < 2)
        throw std::runtime_error("TransposeOperation: tensor must be at least 2D");
    if (dim0_ >= ndim || dim1_ >= ndim)
        throw std::runtime_error("TransposeOperation: dim out of range");

    auto new_shape  = in.shape();
    auto new_stride = in.stride();
    std::swap(new_shape [dim0_], new_shape [dim1_]);
    std::swap(new_stride[dim0_], new_stride[dim1_]);

    return {in.view(std::move(new_shape), std::move(new_stride))};
}

std::vector<Tensor> TransposeOperation::backward(const std::vector<Tensor>& grad_outputs) {
    assert(!grad_outputs.empty() && "TransposeOperation::backward: no grad_outputs");

    const Tensor& grad = grad_outputs[0];

    // Transpose is self-inverse: applying the same swap brings grad back to
    // the input's shape and stride layout.
    auto new_shape  = grad.shape();
    auto new_stride = grad.stride();
    std::swap(new_shape [dim0_], new_shape [dim1_]);
    std::swap(new_stride[dim0_], new_stride[dim1_]);

    return {grad.view(std::move(new_shape), std::move(new_stride))};
}

} // namespace otter
