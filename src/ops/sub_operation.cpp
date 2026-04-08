#include "otter/ops/sub_operation.h"

#include <cassert>

#include "otter/autograd/no_grad_guard.h"
#include "otter/kernel/kernel_engine.h"

namespace otter {

std::vector<Tensor> SubOperation::forward(const std::vector<Tensor>& inputs) {
    assert(inputs.size() == 2 && "SubOperation: requires exactly 2 inputs");
    validate_binary(inputs[0], inputs[1], "Sub");

    Tensor result = Tensor::zeros(inputs[0].shape(), inputs[0].backend(), inputs[0].dtype());
    inputs[0].backend().kernel_engine()->dispatch_binary(
        KernelType::Sub, inputs[0], inputs[1], result);
    return {result};
}

std::vector<Tensor> SubOperation::backward(const std::vector<Tensor>& grad_outputs) {
    // d(a-b)/da = 1  → grad_a = grad_out (pass through)
    // d(a-b)/db = -1 → grad_b = -grad_out
    assert(!grad_outputs.empty() && "SubOperation::backward: no grad_outputs");

    const Tensor& grad = grad_outputs[0];
    NoGradGuard ng;
    Tensor grad_b = Tensor::zeros(grad.shape(), grad.backend(), grad.dtype());
    grad.backend().kernel_engine()->dispatch_unary(KernelType::Neg, grad, grad_b);
    return {grad, grad_b};
}

} // namespace otter
