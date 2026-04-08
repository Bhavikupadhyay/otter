#include "otter/ops/reshape_operation.h"

#include <cassert>
#include <numeric>
#include <stdexcept>

#include "otter/detail/stride_utils.h"

namespace otter {

std::vector<Tensor> ReshapeOperation::forward(const std::vector<Tensor>& inputs) {
    assert(inputs.size() == 1 && "ReshapeOperation: requires exactly 1 input");
    const Tensor& in = inputs[0];

    if (!in.defined())
        throw std::runtime_error("ReshapeOperation: input is undefined");
    if (!in.is_contiguous())
        throw std::runtime_error(
            "ReshapeOperation: input must be contiguous — call contiguous() first");

    const std::size_t in_numel  = in.numel();
    std::size_t       out_numel = 1;
    for (auto d : target_shape_) out_numel *= d;

    if (in_numel != out_numel)
        throw std::runtime_error(
            "ReshapeOperation: numel mismatch (" +
            std::to_string(in_numel) + " → " + std::to_string(out_numel) + ")");

    return {in.view(target_shape_, detail::contiguous_strides(target_shape_))};
}

std::vector<Tensor> ReshapeOperation::backward(const std::vector<Tensor>& grad_outputs) {
    assert(!grad_outputs.empty() && "ReshapeOperation::backward: no grad_outputs");
    assert(saved_inputs_.size() == 1);

    const Tensor& grad       = grad_outputs[0];
    const auto&   orig_shape = saved_inputs_[0].shape();

    // Grad has the output (target) shape. Reshape it back to the original input shape.
    // The grad flowing back from sum/add/matmul is always contiguous, so a raw
    // view is valid. If it somehow is not contiguous, materialize it first.
    if (grad.is_contiguous())
        return {grad.view(orig_shape, detail::contiguous_strides(orig_shape))};

    Tensor grad_c = grad.contiguous();
    return {grad_c.view(orig_shape, detail::contiguous_strides(orig_shape))};
}

} // namespace otter
