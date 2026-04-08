#include "otter/ops/add_operation.h"

#include <cassert>
#include <stdexcept>

#include "otter/kernel/kernel_engine.h"

namespace otter {

std::vector<Tensor> AddOperation::forward(const std::vector<Tensor>& inputs) {
    assert(inputs.size() == 2 && "AddOperation: requires exactly 2 inputs");
    validate_binary(inputs[0], inputs[1], "Add");

    Tensor result = Tensor::zeros(inputs[0].shape(), inputs[0].backend(), inputs[0].dtype());
    inputs[0].backend().kernel_engine()->dispatch_binary(
        KernelType::Add, inputs[0], inputs[1], result);
    return {result};
}

std::vector<Tensor> AddOperation::backward(const std::vector<Tensor>& grad_outputs) {
    // d(a+b)/da = 1, d(a+b)/db = 1 — gradient passes through unchanged.
    // BroadcastOp nodes inserted by execute() reduce grads to original shapes.
    assert(!grad_outputs.empty() && "AddOperation::backward: no grad_outputs");
    return {grad_outputs[0], grad_outputs[0]};
}

} // namespace otter
