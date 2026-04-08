#include "otter/ops/mul_operation.h"

#include <cassert>

#include "otter/autograd/no_grad_guard.h"
#include "otter/kernel/kernel_engine.h"

namespace otter::ops {

std::vector<Tensor> MulOperation::forward(const std::vector<Tensor>& inputs) {
    assert(inputs.size() == 2 && "MulOperation: requires exactly 2 inputs");
    validate_binary(inputs[0], inputs[1], "Mul");

    Tensor result = Tensor::zeros(inputs[0].shape(), inputs[0].backend(), inputs[0].dtype());
    inputs[0].backend().kernel_engine()->dispatch_binary(
        KernelType::Mul, inputs[0], inputs[1], result);
    return {result};
}

std::vector<Tensor> MulOperation::backward(const std::vector<Tensor>& grad_outputs) {
    // Product rule: grad_a = grad_out * b,  grad_b = grad_out * a.
    // saved_inputs_[0] == a (broadcast-adjusted), saved_inputs_[1] == b.
    // Run under NoGradGuard: these multiplications must not build a second-order graph.
    assert(!grad_outputs.empty() && "MulOperation::backward: no grad_outputs");
    assert(saved_inputs_.size() == 2 && "MulOperation::backward: saved_inputs_ must hold a and b");

    const Tensor& grad = grad_outputs[0];
    const Tensor& a    = saved_inputs_[0];
    const Tensor& b    = saved_inputs_[1];

    NoGradGuard ng;
    Tensor grad_a = Tensor::zeros(grad.shape(), grad.backend(), grad.dtype());
    grad.backend().kernel_engine()->dispatch_binary(KernelType::Mul, grad, b, grad_a);

    Tensor grad_b = Tensor::zeros(grad.shape(), grad.backend(), grad.dtype());
    grad.backend().kernel_engine()->dispatch_binary(KernelType::Mul, grad, a, grad_b);

    return {grad_a, grad_b};
}

} // namespace otter::ops
