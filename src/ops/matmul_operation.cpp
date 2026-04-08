#include "otter/ops/matmul_operation.h"

#include <cassert>
#include <stdexcept>

#include "otter/autograd/no_grad_guard.h"
#include "otter/detail/broadcast.h"
#include "otter/kernel/kernel_engine.h"

namespace otter {

// ── Helper: stride-swap transpose of last two dims ────────────────────────────
//
// Returns a view of t with shape[n-1] <-> shape[n-2] and stride[n-1] <-> stride[n-2].
// The CPU matmul kernel reads strides from inputs, so this is sufficient —
// no data movement required.

static Tensor transpose_last2(const Tensor& t) {
    assert(t.shape().size() >= 2 && "transpose_last2: tensor must be at least 2D");
    auto shape  = t.shape();
    auto stride = t.stride();
    const std::size_t n = shape.size();
    std::swap(shape [n - 2], shape [n - 1]);
    std::swap(stride[n - 2], stride[n - 1]);
    return t.view(std::move(shape), std::move(stride));
}


// ── infer_shapes override ─────────────────────────────────────────────────────
//
// A: [..., M, K]   B: [..., K, N]
// 1. Validate ndim >= 2 for both inputs.
// 2. Validate K matches.
// 3. Broadcast batch dims (everything except last 2) via broadcast_shape.
// 4. Return per-input target shapes: {batch + [M, K], batch + [K, N]}.
//    execute() will insert BroadcastOp nodes for any batch dim that expands.

BroadcastSpec MatMulOperation::infer_shapes(const std::vector<Tensor>& inputs) const {
    assert(inputs.size() == 2 && "MatMulOperation::infer_shapes: requires exactly 2 inputs");
    const Tensor& a = inputs[0];
    const Tensor& b = inputs[1];

    if (a.shape().size() < 2)
        throw std::runtime_error("MatMul: A must be at least 2-dimensional");
    if (b.shape().size() < 2)
        throw std::runtime_error("MatMul: B must be at least 2-dimensional");

    const std::size_t an = a.shape().size();
    const std::size_t bn = b.shape().size();
    const std::size_t K_a = a.shape()[an - 1];
    const std::size_t K_b = b.shape()[bn - 2];

    if (K_a != K_b)
        throw std::runtime_error(
            "MatMul: inner dimensions do not match ("
            + std::to_string(K_a) + " vs " + std::to_string(K_b) + ")");

    // Extract batch dims (all but last 2).
    std::vector<std::size_t> a_batch(a.shape().begin(), a.shape().end() - 2);
    std::vector<std::size_t> b_batch(b.shape().begin(), b.shape().end() - 2);

    // Broadcast batch dims. If both have no batch dims, batch = {}.
    std::vector<std::size_t> out_batch;
    if (!a_batch.empty() || !b_batch.empty()) {
        // Pad shorter batch with leading 1s for broadcast_shape compatibility.
        while (a_batch.size() < b_batch.size()) a_batch.insert(a_batch.begin(), 1);
        while (b_batch.size() < a_batch.size()) b_batch.insert(b_batch.begin(), 1);
        out_batch = detail::broadcast_shape(a_batch, b_batch);
    }

    const std::size_t M = a.shape()[an - 2];
    const std::size_t K = K_a;
    const std::size_t N = b.shape()[bn - 1];

    std::vector<std::size_t> target_a = out_batch;
    target_a.push_back(M);
    target_a.push_back(K);

    std::vector<std::size_t> target_b = out_batch;
    target_b.push_back(K);
    target_b.push_back(N);

    BroadcastSpec spec;
    spec.input_target_shapes = {std::move(target_a), std::move(target_b)};
    return spec;
}


// ── forward ───────────────────────────────────────────────────────────────────

std::vector<Tensor> MatMulOperation::forward(const std::vector<Tensor>& inputs) {
    assert(inputs.size() == 2 && "MatMulOperation: requires exactly 2 inputs");
    const Tensor& a = inputs[0];
    const Tensor& b = inputs[1];

    const std::size_t an = a.shape().size();
    const std::size_t bn = b.shape().size();
    const std::size_t M  = a.shape()[an - 2];
    const std::size_t N  = b.shape()[bn - 1];

    // Output shape: broadcast batch dims + [M, N].
    // After infer_shapes + BroadcastOp insertion, a and b have identical batch dims.
    std::vector<std::size_t> out_shape(a.shape().begin(), a.shape().end() - 2);
    out_shape.push_back(M);
    out_shape.push_back(N);

    Tensor result = Tensor::zeros(out_shape, a.backend(), a.dtype());
    a.backend().kernel_engine()->dispatch_matmul(a, b, result);
    return {result};
}


// ── backward ─────────────────────────────────────────────────────────────────
//
// grad: [..., M, N]
// grad_A = grad @ B^T  →  [..., M, N] @ [..., N, K] = [..., M, K]
// grad_B = A^T @ grad  →  [..., K, M] @ [..., M, N] = [..., K, N]

std::vector<Tensor> MatMulOperation::backward(const std::vector<Tensor>& grad_outputs) {
    assert(!grad_outputs.empty() && "MatMulOperation::backward: no grad_outputs");
    assert(saved_inputs_.size() == 2 && "MatMulOperation::backward: saved_inputs_ must hold A and B");

    const Tensor& grad = grad_outputs[0];
    const Tensor& a    = saved_inputs_[0];
    const Tensor& b    = saved_inputs_[1];

    NoGradGuard ng;

    // grad_A = grad @ B^T
    Tensor b_t = transpose_last2(b);
    const std::size_t gn = grad.shape().size();
    const std::size_t M  = grad.shape()[gn - 2];
    const std::size_t K  = b_t.shape()[b_t.shape().size() - 1];

    std::vector<std::size_t> grad_a_shape(grad.shape().begin(), grad.shape().end() - 2);
    grad_a_shape.push_back(M);
    grad_a_shape.push_back(K);
    Tensor grad_a = Tensor::zeros(grad_a_shape, grad.backend(), grad.dtype());
    grad.backend().kernel_engine()->dispatch_matmul(grad, b_t, grad_a);

    // grad_B = A^T @ grad
    Tensor a_t = transpose_last2(a);
    const std::size_t N = grad.shape()[gn - 1];
    const std::size_t K2 = a_t.shape()[a_t.shape().size() - 2];

    std::vector<std::size_t> grad_b_shape(grad.shape().begin(), grad.shape().end() - 2);
    grad_b_shape.push_back(K2);
    grad_b_shape.push_back(N);
    Tensor grad_b = Tensor::zeros(grad_b_shape, grad.backend(), grad.dtype());
    grad.backend().kernel_engine()->dispatch_matmul(a_t, grad, grad_b);

    return {grad_a, grad_b};
}

} // namespace otter
