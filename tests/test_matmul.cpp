#include "test_utils.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "otter/tensor.h"
#include "otter/backends/cpu.h"

// Tests for MatMulOperation: forward values, shape inference, backward gradients,
// batched matmul, batch broadcasting, and chained operations.

namespace otter::test {

// ── Forward: basic 2D matmul ──────────────────────────────────────────────────

void test_matmul_2d_forward() {
    std::cout << "[MatMul 1] 2x3 @ 3x2 — correct output values\n";
    Backend& be = cpu_backend();
    // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
    // C[0,0] = 1*7+2*9+3*11 = 7+18+33 = 58
    // C[0,1] = 1*8+2*10+3*12 = 8+20+36 = 64
    // C[1,0] = 4*7+5*9+6*11 = 28+45+66 = 139
    // C[1,1] = 4*8+5*10+6*12 = 32+50+72 = 154
    Tensor a = Tensor::from_data<double>({1,2,3,4,5,6},       {2, 3}, be);
    Tensor b = Tensor::from_data<double>({7,8,9,10,11,12},    {3, 2}, be);
    Tensor c = a.matmul(b);
    CHECK(c.shape() == (std::vector<std::size_t>{2, 2}));
    CHECK_NEAR(c.at({0, 0}),  58.0, 1e-12);
    CHECK_NEAR(c.at({0, 1}),  64.0, 1e-12);
    CHECK_NEAR(c.at({1, 0}), 139.0, 1e-12);
    CHECK_NEAR(c.at({1, 1}), 154.0, 1e-12);
}

void test_matmul_identity_op() {
    std::cout << "[MatMul 2] 3x3 identity @ A == A\n";
    Backend& be = cpu_backend();
    Tensor I = Tensor::from_data<double>({1,0,0, 0,1,0, 0,0,1}, {3, 3}, be);
    Tensor a = Tensor::from_data<double>({1,2,3, 4,5,6, 7,8,9}, {3, 3}, be);
    Tensor c = I.matmul(a);
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t col = 0; col < 3; ++col)
            CHECK_NEAR(c.at({r, col}), a.at({r, col}), 1e-12);
}

void test_matmul_inner_mismatch_throws() {
    std::cout << "[MatMul 3] inner dim mismatch throws\n";
    Backend& be = cpu_backend();
    Tensor a = Tensor::zeros({2, 3}, be);
    Tensor b = Tensor::zeros({4, 2}, be);
    bool threw = false;
    try { (void)a.matmul(b); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

// ── Backward: 2D grad check ───────────────────────────────────────────────────

void test_matmul_backward_2d() {
    std::cout << "[MatMul 4] backward 2x3 @ 3x2: grad_A = grad@B^T, grad_B = A^T@grad\n";
    Backend& be = cpu_backend();
    // loss = sum(A @ B)
    // d loss / d A = ones_{2x2} @ B^T  (all-ones upstream grad times B transposed)
    // d loss / d B = A^T @ ones_{2x2}
    //
    // A = [[1,2,3],[4,5,6]], B = [[1,0],[0,1],[1,1]]
    // A@B = [[1+0+3, 0+2+3],[4+0+6, 0+5+6]] = [[4,5],[10,11]]
    // sum = 30
    //
    // grad upstream (seed from sum.backward) = [[1,1],[1,1]] (2x2)
    // grad_A = [[1,1],[1,1]] @ [[1,0,1],[0,1,1]] = [[1,2,2],[1,2,2]]  -- but transposed B
    //        = grad @ B^T  where B^T = [[1,0,1],[0,1,1]]
    //          row 0: 1*1+1*0=1,  1*0+1*1=1,  1*1+1*1=2  → [1,1,2]
    //          row 1: same       → [1,1,2]
    // grad_B = A^T @ grad  where A^T = [[1,4],[2,5],[3,6]]
    //          [[1,4],[2,5],[3,6]] @ [[1,1],[1,1]]
    //          row 0: [1+4, 1+4] = [5,5]
    //          row 1: [2+5, 2+5] = [7,7]
    //          row 2: [3+6, 3+6] = [9,9]
    Tensor a = Tensor::from_data<double>({1,2,3, 4,5,6},   {2, 3}, be, /*requires_grad=*/true);
    Tensor b = Tensor::from_data<double>({1,0, 0,1, 1,1},  {3, 2}, be, /*requires_grad=*/true);
    a.matmul(b).sum().backward();

    CHECK_NEAR(a.grad().at({0, 0}), 1.0, 1e-12);
    CHECK_NEAR(a.grad().at({0, 1}), 1.0, 1e-12);
    CHECK_NEAR(a.grad().at({0, 2}), 2.0, 1e-12);
    CHECK_NEAR(a.grad().at({1, 0}), 1.0, 1e-12);
    CHECK_NEAR(a.grad().at({1, 1}), 1.0, 1e-12);
    CHECK_NEAR(a.grad().at({1, 2}), 2.0, 1e-12);

    CHECK_NEAR(b.grad().at({0, 0}), 5.0, 1e-12);
    CHECK_NEAR(b.grad().at({0, 1}), 5.0, 1e-12);
    CHECK_NEAR(b.grad().at({1, 0}), 7.0, 1e-12);
    CHECK_NEAR(b.grad().at({1, 1}), 7.0, 1e-12);
    CHECK_NEAR(b.grad().at({2, 0}), 9.0, 1e-12);
    CHECK_NEAR(b.grad().at({2, 1}), 9.0, 1e-12);
}

// ── Batched matmul ────────────────────────────────────────────────────────────

void test_matmul_batched_forward() {
    std::cout << "[MatMul 5] batched 2x2x3 @ 2x3x2 — output shape {2,2,2}\n";
    Backend& be = cpu_backend();
    // Two independent 2x3 @ 3x2 matmuls stacked in a batch.
    // Batch 0: A0=[[1,0,0],[0,1,0]] B0=[[1,2],[3,4],[5,6]] → [[1,2],[3,4]]
    // Batch 1: A1=[[1,1,1],[2,2,2]] B1=[[1,0],[0,1],[1,1]] → [[2,2],[4,4]]
    Tensor a = Tensor::from_data<double>(
        {1,0,0, 0,1,0,   1,1,1, 2,2,2}, {2, 2, 3}, be);
    Tensor b = Tensor::from_data<double>(
        {1,2, 3,4, 5,6,   1,0, 0,1, 1,1}, {2, 3, 2}, be);
    Tensor c = a.matmul(b);
    CHECK(c.shape() == (std::vector<std::size_t>{2, 2, 2}));
    // Batch 0
    CHECK_NEAR(c.at({0, 0, 0}), 1.0, 1e-12);
    CHECK_NEAR(c.at({0, 0, 1}), 2.0, 1e-12);
    CHECK_NEAR(c.at({0, 1, 0}), 3.0, 1e-12);
    CHECK_NEAR(c.at({0, 1, 1}), 4.0, 1e-12);
    // Batch 1: A1@B1 = [[1+0+1, 0+1+1],[2+0+2, 0+2+2]] = [[2,2],[4,4]]
    CHECK_NEAR(c.at({1, 0, 0}), 2.0, 1e-12);
    CHECK_NEAR(c.at({1, 0, 1}), 2.0, 1e-12);
    CHECK_NEAR(c.at({1, 1, 0}), 4.0, 1e-12);
    CHECK_NEAR(c.at({1, 1, 1}), 4.0, 1e-12);
}

void test_matmul_batch_broadcast() {
    std::cout << "[MatMul 6] batch broadcast {1,2,3} @ {4,3,2} → {4,2,2}\n";
    Backend& be = cpu_backend();
    // A has batch=1, B has batch=4 → BroadcastOp expands A to batch=4.
    Tensor a = Tensor::from_data<double>({1,0,0, 0,1,0}, {1, 2, 3}, be);
    // B: 4 copies of [[1,2],[3,4],[5,6]]
    std::vector<double> b_data;
    for (int i = 0; i < 4; ++i) { for (double v : {1.0,2.0,3.0,4.0,5.0,6.0}) b_data.push_back(v); }
    Tensor b = Tensor::from_data<double>(b_data, {4, 3, 2}, be);
    Tensor c = a.matmul(b);
    CHECK(c.shape() == (std::vector<std::size_t>{4, 2, 2}));
    // Each batch: [[1,0,0],[0,1,0]] @ [[1,2],[3,4],[5,6]] = [[1,2],[3,4]]
    for (std::size_t bi = 0; bi < 4; ++bi) {
        CHECK_NEAR(c.at({bi, 0, 0}), 1.0, 1e-12);
        CHECK_NEAR(c.at({bi, 0, 1}), 2.0, 1e-12);
        CHECK_NEAR(c.at({bi, 1, 0}), 3.0, 1e-12);
        CHECK_NEAR(c.at({bi, 1, 1}), 4.0, 1e-12);
    }
}

// ── Chain: X@W + b (linear layer) ────────────────────────────────────────────

void test_matmul_linear_chain() {
    std::cout << "[MatMul 7] linear: sum((X@W + b)) backward — grad_W, grad_b correct\n";
    Backend& be = cpu_backend();
    // X = [[1,2],[3,4]]  (2x2, no grad — input data)
    // W = [[1,0],[0,1]]  (2x2, requires_grad)
    // b = [[1,1]]        (1x2, requires_grad, broadcasts over rows)
    // loss = sum(X@W + b)
    //      = sum([[1+1, 2+1],[3+1, 4+1]]) = sum([[2,3],[4,5]]) = 14
    // d loss/d W: upstream grad of X@W is ones_{2x2}; grad_W = X^T @ ones = [[4,4],[6,6]]
    // d loss/d b: upstream grad of add is ones_{2x2}; b has shape {1,2}
    //             grad_b sums over batch dim → [[2, 2]]
    Tensor X = Tensor::from_data<double>({1,2,3,4}, {2, 2}, be);
    Tensor W = Tensor::from_data<double>({1,0,0,1}, {2, 2}, be, /*requires_grad=*/true);
    Tensor b = Tensor::from_data<double>({1,1},     {1, 2}, be, /*requires_grad=*/true);
    X.matmul(W).add(b).sum().backward();

    // grad_W = X^T @ ones_{2x2} = [[1,3],[2,4]]^T doesn't matter, just X^T @ [[1,1],[1,1]]
    // = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
    CHECK_NEAR(W.grad().at({0, 0}), 4.0, 1e-12);
    CHECK_NEAR(W.grad().at({0, 1}), 4.0, 1e-12);
    CHECK_NEAR(W.grad().at({1, 0}), 6.0, 1e-12);
    CHECK_NEAR(W.grad().at({1, 1}), 6.0, 1e-12);

    // grad_b: ones_{2x2} reduced over batch dim 0 → [[2,2]]
    CHECK_NEAR(b.grad().at({0, 0}), 2.0, 1e-12);
    CHECK_NEAR(b.grad().at({0, 1}), 2.0, 1e-12);
}

// ── Memory cleanup ────────────────────────────────────────────────────────────

void test_matmul_memory_clean() {
    std::cout << "[MatMul 8] memory: bytes_allocated == 0 after matmul backward\n";
    Backend& be = cpu_backend();
    {
        Tensor a = Tensor::from_data<double>({1,2,3,4,5,6}, {2,3}, be, /*requires_grad=*/true);
        Tensor b = Tensor::from_data<double>({1,0,1,0,1,0}, {3,2}, be, /*requires_grad=*/true);
        a.matmul(b).sum().backward();
    }
    CHECK(be.memory_manager()->bytes_allocated() == 0);
}

// ── Run all ───────────────────────────────────────────────────────────────────

void run_matmul_tests() {
    test_matmul_2d_forward();
    test_matmul_identity_op();
    test_matmul_inner_mismatch_throws();
    test_matmul_backward_2d();
    test_matmul_batched_forward();
    test_matmul_batch_broadcast();
    test_matmul_linear_chain();
    test_matmul_memory_clean();
}

} // namespace otter::test
