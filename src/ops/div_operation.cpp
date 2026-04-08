#include "otter/ops/div_operation.h"

#include <cassert>

#include "otter/autograd/no_grad_guard.h"
#include "otter/kernel/kernel_engine.h"

namespace otter::ops {

std::vector<Tensor> DivOperation::forward(const std::vector<Tensor>& inputs) {
    assert(inputs.size() == 2 && "DivOperation: requires exactly 2 inputs");
    validate_binary(inputs[0], inputs[1], "Div");

    Tensor result = Tensor::zeros(inputs[0].shape(), inputs[0].backend(), inputs[0].dtype());
    inputs[0].backend().kernel_engine()->dispatch_binary(
        KernelType::Div, inputs[0], inputs[1], result);
    return {result};
}

std::vector<Tensor> DivOperation::backward(const std::vector<Tensor>& grad_outputs) {
    // grad_a = grad_out / b
    // grad_b = -grad_out * a / (b * b)
    // IEEE 754: b=0 yields ±inf or nan in both — no clamping, no checks.
    assert(!grad_outputs.empty() && "DivOperation::backward: no grad_outputs");
    assert(saved_inputs_.size() == 2 && "DivOperation::backward: saved_inputs_ must hold a and b");

    const Tensor& grad = grad_outputs[0];
    const Tensor& a    = saved_inputs_[0];
    const Tensor& b    = saved_inputs_[1];
    auto* ke = grad.backend().kernel_engine();

    NoGradGuard ng;

    // grad_a = grad_out / b
    Tensor grad_a = Tensor::zeros(grad.shape(), grad.backend(), grad.dtype());
    ke->dispatch_binary(KernelType::Div, grad, b, grad_a);

    // grad_b = -(grad_out * a) / (b * b)
    Tensor tmp1   = Tensor::zeros(grad.shape(), grad.backend(), grad.dtype());
    ke->dispatch_binary(KernelType::Mul, grad, a, tmp1);       // grad * a

    Tensor tmp2   = Tensor::zeros(b.shape(), b.backend(), b.dtype());
    ke->dispatch_binary(KernelType::Mul, b, b, tmp2);          // b * b

    Tensor tmp3   = Tensor::zeros(grad.shape(), grad.backend(), grad.dtype());
    ke->dispatch_binary(KernelType::Div, tmp1, tmp2, tmp3);    // (grad * a) / (b * b)

    Tensor grad_b = Tensor::zeros(grad.shape(), grad.backend(), grad.dtype());
    ke->dispatch_unary(KernelType::Neg, tmp3, grad_b);         // negate

    return {grad_a, grad_b};
}

} // namespace otter::ops
