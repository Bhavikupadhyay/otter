#include "otter/ops/sum_operation.h"

#include <cassert>
#include <stdexcept>

#include "otter/kernel/kernel_engine.h"

namespace otter::ops {

std::vector<Tensor> SumOperation::forward(const std::vector<Tensor>& inputs) {
    assert(inputs.size() == 1 && "SumOperation: requires exactly 1 input");
    if (!inputs[0].defined())
        throw std::runtime_error("Sum: input Tensor is undefined");

    Tensor result = Tensor::zeros({1}, inputs[0].backend(), inputs[0].dtype());
    inputs[0].backend().kernel_engine()->dispatch_unary(KernelType::ReduceSum, inputs[0], result);
    return {result};
}

std::vector<Tensor> SumOperation::backward(const std::vector<Tensor>& grad_outputs) {
    // d(sum(x))/dx_i = 1 for all i.  Fan the scalar gradient out to input shape.
    assert(!grad_outputs.empty() && "SumOperation::backward: no grad_outputs");
    assert(!saved_inputs_.empty() && "SumOperation::backward: no saved inputs");

    // Read scalar gradient from the output (shape {1}).
    // On a GPU backend this forces a device→host sync — see TODO in prototype.
    const double scalar_grad = grad_outputs[0].at({0});

    Tensor grad_input = Tensor::zeros(
        saved_inputs_[0].shape(),
        saved_inputs_[0].backend(),
        saved_inputs_[0].dtype());
    grad_input.fill_(scalar_grad);
    return {grad_input};
}

} // namespace otter::ops
