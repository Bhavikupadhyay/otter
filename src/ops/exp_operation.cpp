#include "otter/ops/exp_operation.h"

#include <cassert>
#include <stdexcept>

#include "otter/autograd/no_grad_guard.h"
#include "otter/kernel/kernel_engine.h"

namespace otter {

std::vector<Tensor> ExpOperation::forward(const std::vector<Tensor>& inputs) {
    assert(inputs.size() == 1 && "ExpOperation: requires exactly 1 input");
    if (!inputs[0].defined())
        throw std::runtime_error("Exp: input Tensor is undefined");

    Tensor result = Tensor::zeros(inputs[0].shape(), inputs[0].backend(), inputs[0].dtype());
    inputs[0].backend().kernel_engine()->dispatch_unary(KernelType::Exp, inputs[0], result);
    return {result};
}

std::vector<Tensor> ExpOperation::backward(const std::vector<Tensor>& grad_outputs) {
    // d(exp(x))/dx = exp(x) — reuse saved output to avoid recomputation.
    assert(!grad_outputs.empty() && "ExpOperation::backward: no grad_outputs");
    assert(!saved_outputs_.empty() && "ExpOperation::backward: saved_outputs_ is empty");

    const Tensor& grad    = grad_outputs[0];
    const Tensor& exp_out = saved_outputs_[0];

    NoGradGuard ng;
    Tensor result = Tensor::zeros(grad.shape(), grad.backend(), grad.dtype());
    grad.backend().kernel_engine()->dispatch_binary(KernelType::Mul, grad, exp_out, result);
    return {result};
}

} // namespace otter
