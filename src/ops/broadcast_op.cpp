#include "broadcast_op.h"

#include <cassert>
#include <stdexcept>

#include "otter/detail/broadcast.h"
#include "otter/kernel/kernel_engine.h"

namespace otter {

std::vector<Tensor> BroadcastOp::forward(const std::vector<Tensor>& inputs) {
    assert(inputs.size() == 1 && "BroadcastOp: takes exactly one input");
    if (!inputs[0].defined())
        throw std::runtime_error("BroadcastOp: input is undefined");

    // Validate compatibility via broadcast_shape (throws runtime_error on
    // incompatible shapes). broadcast_strides asserts compatibility, so
    // this check must come first to give a clean error instead of a crash.
    detail::broadcast_shape(inputs[0].shape(), target_shape_);

    const auto new_strides = detail::broadcast_strides(
        inputs[0].shape(), inputs[0].stride(), target_shape_);
    // Stride-zero view: shares buffer, no data copy, O(1).
    return {inputs[0].view(target_shape_, new_strides)};
}

std::vector<Tensor> BroadcastOp::backward(const std::vector<Tensor>& grad_outputs) {
    // grad_outputs[0].shape() == target_shape_ (broadcast shape)
    // saved_inputs_[0].shape() == original pre-broadcast shape
    // Reduce grad back to original shape by summing over broadcast axes.
    assert(!saved_inputs_.empty() && "BroadcastOp::backward: no saved inputs");

    const Tensor& grad   = grad_outputs[0];
    const Tensor& orig   = saved_inputs_[0];
    Tensor grad_input = Tensor::zeros(orig.shape(), grad.backend(), grad.dtype());
    // dispatch_unary(ReduceTo, src, dst): accumulates src (broadcast shape) into
    // dst (original shape) by summing over broadcast dimensions.
    grad.backend().kernel_engine()->dispatch_unary(KernelType::ReduceTo, grad, grad_input);
    return {grad_input};
}

} // namespace otter
