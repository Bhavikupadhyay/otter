#include "otter/ops/slice_operation.h"

#include <cassert>
#include <stdexcept>

namespace otter {

std::vector<Tensor> SliceOperation::forward(const std::vector<Tensor>& inputs) {
    assert(inputs.size() == 1 && "SliceOperation: requires exactly 1 input");
    const Tensor& in = inputs[0];

    if (!in.defined())
        throw std::runtime_error("SliceOperation: input is undefined");

    const std::size_t ndim = in.shape().size();
    if (dim_ >= ndim)
        throw std::runtime_error("SliceOperation: dim out of range");
    if (start_ + length_ > in.shape()[dim_])
        throw std::runtime_error(
            "SliceOperation: start(" + std::to_string(start_) +
            ") + length(" + std::to_string(length_) +
            ") exceeds dim size(" + std::to_string(in.shape()[dim_]) + ")");

    auto new_shape  = in.shape();
    new_shape[dim_] = length_;

    // Offset advances by start * stride[dim]; strides are unchanged.
    const std::size_t new_offset = in.offset() + start_ * in.stride()[dim_];

    return {in.view(std::move(new_shape), in.stride(), new_offset)};
}

std::vector<Tensor> SliceOperation::backward(const std::vector<Tensor>& grad_outputs) {
    assert(!grad_outputs.empty() && "SliceOperation::backward: no grad_outputs");
    assert(saved_inputs_.size() == 1);

    const Tensor& grad = grad_outputs[0];
    const Tensor& orig = saved_inputs_[0];

    // Allocate a zeros tensor with the original shape.
    Tensor grad_input = Tensor::zeros(orig.shape(), grad.backend(), grad.dtype());

    // Create a view of grad_input at the same slice position as the forward pass.
    // The view has the same shape as grad and points into grad_input's buffer.
    const std::size_t slice_offset =
        grad_input.offset() + start_ * grad_input.stride()[dim_];

    auto slice_shape  = grad.shape();                           // = orig shape with dim_=length_
    auto slice_stride = grad_input.stride();                    // contiguous strides of full tensor

    Tensor grad_input_slice =
        grad_input.view(std::move(slice_shape), slice_stride, slice_offset);

    // Copy grad into the slice region of grad_input (strided dst supported since
    // cpu_copy was generalised to handle non-contiguous destinations).
    grad.backend().kernel_engine()->dispatch_copy(grad, grad_input_slice);

    return {grad_input};
}

} // namespace otter
