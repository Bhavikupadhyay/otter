#include "otter/ops/sqrt_operation.h"

#include <cassert>
#include <stdexcept>

#include "otter/autograd/no_grad_guard.h"
#include "otter/kernel/kernel_engine.h"

namespace otter::ops {

std::vector<Tensor> SqrtOperation::forward(const std::vector<Tensor>& inputs) {
    assert(inputs.size() == 1 && "SqrtOperation: requires exactly 1 input");
    if (!inputs[0].defined())
        throw std::runtime_error("Sqrt: input Tensor is undefined");

    Tensor result = Tensor::zeros(inputs[0].shape(), inputs[0].backend(), inputs[0].dtype());
    inputs[0].backend().kernel_engine()->dispatch_unary(KernelType::Sqrt, inputs[0], result);
    return {result};
}

std::vector<Tensor> SqrtOperation::backward(const std::vector<Tensor>& grad_outputs) {
    // d(sqrt(x))/dx = 1 / (2 * sqrt(x))  →  grad_x = grad_out / (2 * sqrt(x)).
    // sqrt(x) is the forward output — reuse saved_outputs_[0].
    // IEEE 754: x=0 → saved_out=0 → grad = +inf; x<0 → saved_out=nan → grad=nan.
    assert(!grad_outputs.empty() && "SqrtOperation::backward: no grad_outputs");
    assert(!saved_outputs_.empty() && "SqrtOperation::backward: saved_outputs_ is empty");

    const Tensor& grad     = grad_outputs[0];
    const Tensor& sqrt_out = saved_outputs_[0];
    auto* ke = grad.backend().kernel_engine();

    NoGradGuard ng;

    // two_sqrt = 2.0 * sqrt(x): fill a tensor with 2.0 then multiply element-wise.
    Tensor two    = Tensor::zeros(sqrt_out.shape(), sqrt_out.backend(), sqrt_out.dtype());
    ke->dispatch_fill(two, 2.0);

    Tensor two_sqrt = Tensor::zeros(sqrt_out.shape(), sqrt_out.backend(), sqrt_out.dtype());
    ke->dispatch_binary(KernelType::Mul, two, sqrt_out, two_sqrt);

    Tensor result = Tensor::zeros(grad.shape(), grad.backend(), grad.dtype());
    ke->dispatch_binary(KernelType::Div, grad, two_sqrt, result);
    return {result};
}

} // namespace otter::ops
