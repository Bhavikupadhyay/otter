#include "otter/ops/relu_operation.h"

#include <cassert>
#include <stdexcept>

#include "otter/autograd/no_grad_guard.h"
#include "otter/kernel/kernel_engine.h"

namespace otter {

std::vector<Tensor> ReluOperation::forward(const std::vector<Tensor>& inputs) {
    assert(inputs.size() == 1 && "ReluOperation: requires exactly 1 input");
    if (!inputs[0].defined())
        throw std::runtime_error("Relu: input Tensor is undefined");

    Tensor result = Tensor::zeros(inputs[0].shape(), inputs[0].backend(), inputs[0].dtype());
    inputs[0].backend().kernel_engine()->dispatch_unary(KernelType::Relu, inputs[0], result);
    return {result};
}

std::vector<Tensor> ReluOperation::backward(const std::vector<Tensor>& grad_outputs) {
    // grad_x = grad_out * mask,  where mask[i] = (x[i] > 0) ? 1.0 : 0.0.
    // At x=0: mask = 0.0 (right-hand derivative, matches PyTorch convention).
    assert(!grad_outputs.empty() && "ReluOperation::backward: no grad_outputs");
    assert(!saved_inputs_.empty() && "ReluOperation::backward: saved_inputs_ is empty");

    const Tensor& grad = grad_outputs[0];
    const Tensor& x    = saved_inputs_[0];
    auto* ke = grad.backend().kernel_engine();

    NoGradGuard ng;

    Tensor mask   = Tensor::zeros(x.shape(), x.backend(), x.dtype());
    ke->dispatch_unary(KernelType::ReluMask, x, mask);

    Tensor result = Tensor::zeros(grad.shape(), grad.backend(), grad.dtype());
    ke->dispatch_binary(KernelType::Mul, grad, mask, result);
    return {result};
}

} // namespace otter
