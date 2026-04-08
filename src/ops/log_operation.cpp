#include "otter/ops/log_operation.h"

#include <cassert>
#include <stdexcept>

#include "otter/autograd/no_grad_guard.h"
#include "otter/kernel/kernel_engine.h"

namespace otter::ops {

std::vector<Tensor> LogOperation::forward(const std::vector<Tensor>& inputs) {
    assert(inputs.size() == 1 && "LogOperation: requires exactly 1 input");
    if (!inputs[0].defined())
        throw std::runtime_error("Log: input Tensor is undefined");

    Tensor result = Tensor::zeros(inputs[0].shape(), inputs[0].backend(), inputs[0].dtype());
    inputs[0].backend().kernel_engine()->dispatch_unary(KernelType::Log, inputs[0], result);
    return {result};
}

std::vector<Tensor> LogOperation::backward(const std::vector<Tensor>& grad_outputs) {
    // d(log(x))/dx = 1/x → grad_x = grad_out / x.
    // IEEE 754: x=0 → grad = ±inf; x<0 → grad = nan.
    assert(!grad_outputs.empty() && "LogOperation::backward: no grad_outputs");
    assert(!saved_inputs_.empty() && "LogOperation::backward: saved_inputs_ is empty");

    const Tensor& grad = grad_outputs[0];
    const Tensor& x    = saved_inputs_[0];

    NoGradGuard ng;
    Tensor result = Tensor::zeros(grad.shape(), grad.backend(), grad.dtype());
    grad.backend().kernel_engine()->dispatch_binary(KernelType::Div, grad, x, result);
    return {result};
}

} // namespace otter::ops
