#include "otter/ops/neg_operation.h"

#include <cassert>
#include <stdexcept>

#include "otter/autograd/no_grad_guard.h"
#include "otter/kernel/kernel_engine.h"

namespace otter {

std::vector<Tensor> NegOperation::forward(const std::vector<Tensor>& inputs) {
    assert(inputs.size() == 1 && "NegOperation: requires exactly 1 input");
    if (!inputs[0].defined())
        throw std::runtime_error("Neg: input Tensor is undefined");

    Tensor result = Tensor::zeros(inputs[0].shape(), inputs[0].backend(), inputs[0].dtype());
    inputs[0].backend().kernel_engine()->dispatch_unary(KernelType::Neg, inputs[0], result);
    return {result};
}

std::vector<Tensor> NegOperation::backward(const std::vector<Tensor>& grad_outputs) {
    // d(-x)/dx = -1 — negate the incoming gradient.
    assert(!grad_outputs.empty() && "NegOperation::backward: no grad_outputs");

    const Tensor& grad = grad_outputs[0];
    NoGradGuard ng;
    Tensor neg_grad = Tensor::zeros(grad.shape(), grad.backend(), grad.dtype());
    grad.backend().kernel_engine()->dispatch_unary(KernelType::Neg, grad, neg_grad);
    return {neg_grad};
}

} // namespace otter
