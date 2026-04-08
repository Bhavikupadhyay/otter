#include "otter/ops/mean_operation.h"

#include <cassert>
#include <stdexcept>

#include "otter/kernel/kernel_engine.h"

namespace otter::ops {

std::vector<Tensor> MeanOperation::forward(const std::vector<Tensor>& inputs) {
    assert(inputs.size() == 1 && "MeanOperation: requires exactly 1 input");
    if (!inputs[0].defined())
        throw std::runtime_error("Mean: input Tensor is undefined");

    // contiguous() is a no-op for already-contiguous inputs.
    Tensor src = inputs[0].contiguous();
    auto* ke   = src.backend().kernel_engine();
    const std::size_t n = src.numel();

    // Sum all elements → shape {1}.
    Tensor sum_out = Tensor::zeros({1}, src.backend(), src.dtype());
    ke->dispatch_unary(KernelType::ReduceSum, src, sum_out);

    // Divide sum by n entirely on-device (no host sync in forward).
    Tensor n_tensor = Tensor::zeros({1}, src.backend(), src.dtype());
    n_tensor.fill_(static_cast<double>(n));

    Tensor result = Tensor::zeros({1}, src.backend(), src.dtype());
    ke->dispatch_binary(KernelType::Div, sum_out, n_tensor, result);
    return {result};
}

std::vector<Tensor> MeanOperation::backward(const std::vector<Tensor>& grad_outputs) {
    // d(mean(x))/dx_i = 1/n for all i → fan scalar_grad / n out to input shape.
    assert(!grad_outputs.empty() && "MeanOperation::backward: no grad_outputs");
    assert(!saved_inputs_.empty() && "MeanOperation::backward: no saved inputs");

    const double scalar_grad = grad_outputs[0].at({0});
    const std::size_t n = saved_inputs_[0].numel();

    Tensor grad_input = Tensor::zeros(
        saved_inputs_[0].shape(),
        saved_inputs_[0].backend(),
        saved_inputs_[0].dtype());
    grad_input.fill_(scalar_grad / static_cast<double>(n));
    return {grad_input};
}

} // namespace otter::ops
