#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "../utils/test_utils.h"
#include "otter/tensor.h"
#include "otter/kernel/backend.h"

namespace otter::test::shared {

// ── add ───────────────────────────────────────────────────────────────────────

inline void test_add_forward_values(Backend& be) {
    std::cout << "[OpsForward 1] add: {2,2} + {2,2} — correct values\n";
    // a = [[1,2],[3,4]], b = [[10,20],[30,40]]
    Tensor a = Tensor::from_data<double>({1, 2, 3, 4},       {2, 2}, be);
    Tensor b = Tensor::from_data<double>({10, 20, 30, 40},   {2, 2}, be);
    Tensor c = a.add(b);
    CHECK_NEAR(c.at({0, 0}), 11.0, 1e-12);
    CHECK_NEAR(c.at({0, 1}), 22.0, 1e-12);
    CHECK_NEAR(c.at({1, 0}), 33.0, 1e-12);
    CHECK_NEAR(c.at({1, 1}), 44.0, 1e-12);
}

inline void test_add_output_not_leaf(Backend& be) {
    std::cout << "[OpsForward 2] add: output is non-leaf when any input requires_grad\n";
    Tensor a = Tensor::from_data<double>({1, 2}, {2}, be, /*requires_grad=*/true);
    Tensor b = Tensor::from_data<double>({3, 4}, {2}, be, /*requires_grad=*/false);
    Tensor c = a.add(b);
    CHECK(c.requires_grad());
    CHECK(!c.is_leaf());
}

inline void test_add_no_grad_when_inputs_detached(Backend& be) {
    std::cout << "[OpsForward 3] add: no grad graph when both inputs have requires_grad=false\n";
    Tensor a = Tensor::from_data<double>({1, 2}, {2}, be);
    Tensor b = Tensor::from_data<double>({3, 4}, {2}, be);
    Tensor c = a.add(b);
    CHECK(!c.requires_grad());
    CHECK(c.is_leaf());
}

// ── sum ───────────────────────────────────────────────────────────────────────

inline void test_sum_forward_value(Backend& be) {
    std::cout << "[OpsForward 4] sum: {2,3} → scalar == 21\n";
    Tensor a = Tensor::from_data<double>({1, 2, 3, 4, 5, 6}, {2, 3}, be);
    Tensor s = a.sum();
    CHECK(s.shape() == std::vector<std::size_t>{1});
    CHECK_NEAR(s.at({0}), 21.0, 1e-12);
}

// ── mul ───────────────────────────────────────────────────────────────────────

inline void test_mul_forward_values(Backend& be) {
    std::cout << "[OpsForward 5] mul: {2,2} * {2,2} — correct values\n";
    Tensor a = Tensor::from_data<double>({1, 2, 3, 4},     {2, 2}, be);
    Tensor b = Tensor::from_data<double>({10, 20, 30, 40}, {2, 2}, be);
    Tensor c = a.mul(b);
    CHECK_NEAR(c.at({0, 0}),  10.0, 1e-12);
    CHECK_NEAR(c.at({0, 1}),  40.0, 1e-12);
    CHECK_NEAR(c.at({1, 0}),  90.0, 1e-12);
    CHECK_NEAR(c.at({1, 1}), 160.0, 1e-12);
}

// ── neg ───────────────────────────────────────────────────────────────────────

inline void test_neg_forward(Backend& be) {
    std::cout << "[OpsForward 6] neg: forward values and shape\n";
    Tensor a = Tensor::from_data<double>({-1.0, 2.0, -3.0}, {3}, be);
    Tensor b = a.neg();
    CHECK(b.shape() == (std::vector<std::size_t>{3}));
    CHECK_NEAR(b.at({0}),  1.0, 1e-12);  // -(-1) = 1
    CHECK_NEAR(b.at({1}), -2.0, 1e-12);  // -(2) = -2
    CHECK_NEAR(b.at({2}),  3.0, 1e-12);  // -(-3) = 3
}

inline void test_neg_no_grad_leaf(Backend& be) {
    std::cout << "[OpsForward 7] neg: no grad graph when input has requires_grad=false\n";
    Tensor a = Tensor::from_data<double>({1.0, 2.0}, {2}, be);
    Tensor b = a.neg();
    CHECK(!a.requires_grad());
    CHECK(b.is_leaf());
}

// ── sub ───────────────────────────────────────────────────────────────────────

inline void test_sub_forward(Backend& be) {
    std::cout << "[OpsForward 8] sub: forward values\n";
    // [5, 3, 1] - [1, 2, 3] = [4, 1, -2]
    Tensor a = Tensor::from_data<double>({5.0, 3.0, 1.0}, {3}, be);
    Tensor b = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be);
    Tensor c = a.sub(b);
    CHECK(c.shape() == (std::vector<std::size_t>{3}));
    CHECK_NEAR(c.at({0}),  4.0, 1e-12);
    CHECK_NEAR(c.at({1}),  1.0, 1e-12);
    CHECK_NEAR(c.at({2}), -2.0, 1e-12);
}

// ── div ───────────────────────────────────────────────────────────────────────

inline void test_div_forward(Backend& be) {
    std::cout << "[OpsForward 9] div: forward values\n";
    // [6, 8, 9] / [2, 4, 3] = [3, 2, 3]
    Tensor a = Tensor::from_data<double>({6.0, 8.0, 9.0}, {3}, be);
    Tensor b = Tensor::from_data<double>({2.0, 4.0, 3.0}, {3}, be);
    Tensor c = a.div(b);
    CHECK_NEAR(c.at({0}), 3.0, 1e-12);
    CHECK_NEAR(c.at({1}), 2.0, 1e-12);
    CHECK_NEAR(c.at({2}), 3.0, 1e-12);
}

inline void test_div_ieee754_edge_cases(Backend& be) {
    std::cout << "[OpsForward 10] div: IEEE 754 edge cases — div by zero\n";
    Tensor a1 = Tensor::from_data<double>({ 1.0}, {1}, be);
    Tensor b0 = Tensor::from_data<double>({ 0.0}, {1}, be);
    Tensor r1 = a1.div(b0);
    CHECK(std::isinf(r1.at({0})) && r1.at({0}) > 0.0);  // +inf

    Tensor am = Tensor::from_data<double>({-1.0}, {1}, be);
    Tensor r2 = am.div(b0);
    CHECK(std::isinf(r2.at({0})) && r2.at({0}) < 0.0);  // -inf

    Tensor a0 = Tensor::from_data<double>({ 0.0}, {1}, be);
    Tensor r3 = a0.div(b0);
    CHECK(std::isnan(r3.at({0})));  // nan
}

// ── exp ───────────────────────────────────────────────────────────────────────

inline void test_exp_forward(Backend& be) {
    std::cout << "[OpsForward 11] exp: forward values\n";
    // exp(0) = 1, exp(1) ≈ 2.71828
    Tensor a = Tensor::from_data<double>({0.0, 1.0}, {2}, be);
    Tensor b = a.exp();
    CHECK_NEAR(b.at({0}), 1.0,           1e-12);
    CHECK_NEAR(b.at({1}), std::exp(1.0), 1e-12);
}

inline void test_exp_ieee754_edge_cases(Backend& be) {
    std::cout << "[OpsForward 12] exp: IEEE 754 edge cases — overflow and underflow\n";
    Tensor hi  = Tensor::from_data<double>({ 1000.0}, {1}, be);
    Tensor rhi = hi.exp();
    CHECK(std::isinf(rhi.at({0})) && rhi.at({0}) > 0.0);  // +inf

    Tensor lo  = Tensor::from_data<double>({-1000.0}, {1}, be);
    Tensor rlo = lo.exp();
    CHECK_NEAR(rlo.at({0}), 0.0, 1e-300);  // underflow
}

// ── log ───────────────────────────────────────────────────────────────────────

inline void test_log_forward(Backend& be) {
    std::cout << "[OpsForward 13] log: forward values\n";
    // log(1) = 0, log(e) = 1
    Tensor a = Tensor::from_data<double>({1.0, std::exp(1.0)}, {2}, be);
    Tensor b = a.log();
    CHECK_NEAR(b.at({0}), 0.0, 1e-12);
    CHECK_NEAR(b.at({1}), 1.0, 1e-12);
}

inline void test_log_ieee754_edge_cases(Backend& be) {
    std::cout << "[OpsForward 14] log: IEEE 754 — log(0)=-inf, log(-1)=nan\n";
    Tensor z   = Tensor::from_data<double>({0.0},  {1}, be);
    Tensor rz  = z.log();
    CHECK(std::isinf(rz.at({0})) && rz.at({0}) < 0.0);  // -inf

    Tensor neg  = Tensor::from_data<double>({-1.0}, {1}, be);
    Tensor rneg = neg.log();
    CHECK(std::isnan(rneg.at({0})));  // nan
}

// ── sqrt ─────────────────────────────────────────────────────────────────────

inline void test_sqrt_forward(Backend& be) {
    std::cout << "[OpsForward 15] sqrt: forward values\n";
    // sqrt(4) = 2, sqrt(9) = 3
    Tensor a = Tensor::from_data<double>({4.0, 9.0}, {2}, be);
    Tensor b = a.sqrt();
    CHECK_NEAR(b.at({0}), 2.0, 1e-12);
    CHECK_NEAR(b.at({1}), 3.0, 1e-12);
}

inline void test_sqrt_ieee754_edge_cases(Backend& be) {
    std::cout << "[OpsForward 16] sqrt: IEEE 754 — sqrt(-1)=nan, sqrt(0)=0\n";
    Tensor neg  = Tensor::from_data<double>({-1.0}, {1}, be);
    Tensor rneg = neg.sqrt();
    CHECK(std::isnan(rneg.at({0})));  // nan

    Tensor z  = Tensor::from_data<double>({0.0}, {1}, be);
    Tensor rz = z.sqrt();
    CHECK_NEAR(rz.at({0}), 0.0, 1e-12);
}

// ── relu ─────────────────────────────────────────────────────────────────────

inline void test_relu_forward(Backend& be) {
    std::cout << "[OpsForward 17] relu: positive passes, negative zeros\n";
    Tensor a = Tensor::from_data<double>({3.0, -5.0, 0.0}, {3}, be);
    Tensor b = a.relu();
    CHECK_NEAR(b.at({0}), 3.0, 1e-12);  // positive: pass through
    CHECK_NEAR(b.at({1}), 0.0, 1e-12);  // negative: zero
    CHECK_NEAR(b.at({2}), 0.0, 1e-12);  // zero: zero (right-hand derivative)
}

// ── graph properties ─────────────────────────────────────────────────────────

inline void test_math_ops_graph_properties(Backend& be) {
    std::cout << "[OpsForward 18] graph: is_leaf false for op results with requires_grad input\n";
    Tensor a = Tensor::from_data<double>({1.0, 2.0}, {2}, be, true);
    Tensor b = Tensor::from_data<double>({2.0, 3.0}, {2}, be, true);
    CHECK(!a.sub(b).is_leaf());
    CHECK(!a.div(b).is_leaf());
    CHECK(!a.neg().is_leaf());
    CHECK(!a.exp().is_leaf());
    CHECK(!a.log().is_leaf());
    Tensor pos = Tensor::from_data<double>({1.0, 4.0}, {2}, be, true);
    CHECK(!pos.sqrt().is_leaf());
    CHECK(!pos.relu().is_leaf());
}

inline void test_math_ops_no_grad_leaf(Backend& be) {
    std::cout << "[OpsForward 19] graph: no grad graph built when no input requires_grad\n";
    Tensor a = Tensor::from_data<double>({1.0, 2.0}, {2}, be);
    Tensor b = Tensor::from_data<double>({3.0, 4.0}, {2}, be);
    CHECK(a.sub(b).is_leaf());
    CHECK(a.exp().is_leaf());
}

// ── matmul ───────────────────────────────────────────────────────────────────

inline void test_matmul_2d_forward(Backend& be) {
    std::cout << "[OpsForward 20] matmul: 2x3 @ 3x2 — correct output values\n";
    // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
    // C[0,0]=58, C[0,1]=64, C[1,0]=139, C[1,1]=154
    Tensor a = Tensor::from_data<double>({1,2,3,4,5,6},    {2, 3}, be);
    Tensor b = Tensor::from_data<double>({7,8,9,10,11,12}, {3, 2}, be);
    Tensor c = a.matmul(b);
    CHECK(c.shape() == (std::vector<std::size_t>{2, 2}));
    CHECK_NEAR(c.at({0, 0}),  58.0, 1e-12);
    CHECK_NEAR(c.at({0, 1}),  64.0, 1e-12);
    CHECK_NEAR(c.at({1, 0}), 139.0, 1e-12);
    CHECK_NEAR(c.at({1, 1}), 154.0, 1e-12);
}

inline void test_matmul_identity_op(Backend& be) {
    std::cout << "[OpsForward 21] matmul: 3x3 identity @ A == A\n";
    Tensor I = Tensor::from_data<double>({1,0,0, 0,1,0, 0,0,1}, {3, 3}, be);
    Tensor a = Tensor::from_data<double>({1,2,3, 4,5,6, 7,8,9}, {3, 3}, be);
    Tensor c = I.matmul(a);
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t col = 0; col < 3; ++col)
            CHECK_NEAR(c.at({r, col}), a.at({r, col}), 1e-12);
}

inline void test_matmul_inner_mismatch_throws(Backend& be) {
    std::cout << "[OpsForward 22] matmul: inner dim mismatch throws\n";
    Tensor a = Tensor::zeros({2, 3}, be);
    Tensor b = Tensor::zeros({4, 2}, be);
    bool threw = false;
    try { (void)a.matmul(b); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

inline void test_matmul_batched_forward(Backend& be) {
    std::cout << "[OpsForward 23] matmul: batched 2x2x3 @ 2x3x2 — output shape {2,2,2}\n";
    // Batch 0: [[1,0,0],[0,1,0]] @ [[1,2],[3,4],[5,6]] = [[1,2],[3,4]]
    // Batch 1: [[1,1,1],[2,2,2]] @ [[1,0],[0,1],[1,1]] = [[2,2],[4,4]]
    Tensor a = Tensor::from_data<double>(
        {1,0,0, 0,1,0,   1,1,1, 2,2,2}, {2, 2, 3}, be);
    Tensor b = Tensor::from_data<double>(
        {1,2, 3,4, 5,6,   1,0, 0,1, 1,1}, {2, 3, 2}, be);
    Tensor c = a.matmul(b);
    CHECK(c.shape() == (std::vector<std::size_t>{2, 2, 2}));
    CHECK_NEAR(c.at({0, 0, 0}), 1.0, 1e-12);
    CHECK_NEAR(c.at({0, 0, 1}), 2.0, 1e-12);
    CHECK_NEAR(c.at({0, 1, 0}), 3.0, 1e-12);
    CHECK_NEAR(c.at({0, 1, 1}), 4.0, 1e-12);
    CHECK_NEAR(c.at({1, 0, 0}), 2.0, 1e-12);
    CHECK_NEAR(c.at({1, 0, 1}), 2.0, 1e-12);
    CHECK_NEAR(c.at({1, 1, 0}), 4.0, 1e-12);
    CHECK_NEAR(c.at({1, 1, 1}), 4.0, 1e-12);
}

inline void test_matmul_batch_broadcast(Backend& be) {
    std::cout << "[OpsForward 24] matmul: batch broadcast {1,2,3} @ {4,3,2} → {4,2,2}\n";
    Tensor a = Tensor::from_data<double>({1,0,0, 0,1,0}, {1, 2, 3}, be);
    std::vector<double> b_data;
    for (int i = 0; i < 4; ++i)
        for (double v : {1.0,2.0,3.0,4.0,5.0,6.0}) b_data.push_back(v);
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

inline void test_matmul_linear_chain(Backend& be) {
    std::cout << "[OpsForward 25] linear: sum((X@W + b)) backward — grad_W, grad_b correct\n";
    // X={2x2}, W={2x2} (requires_grad), b={1x2} (requires_grad)
    // loss = sum(X@W + b) = 14
    // grad_W = X^T @ ones_{2x2} = [[4,4],[6,6]]
    // grad_b = ones_{2x2} reduced over batch → [[2,2]]
    Tensor X = Tensor::from_data<double>({1,2,3,4}, {2, 2}, be);
    Tensor W = Tensor::from_data<double>({1,0,0,1}, {2, 2}, be, /*requires_grad=*/true);
    Tensor b = Tensor::from_data<double>({1,1},     {1, 2}, be, /*requires_grad=*/true);
    X.matmul(W).add(b).sum().backward();
    CHECK_NEAR(W.grad().at({0, 0}), 4.0, 1e-12);
    CHECK_NEAR(W.grad().at({0, 1}), 4.0, 1e-12);
    CHECK_NEAR(W.grad().at({1, 0}), 6.0, 1e-12);
    CHECK_NEAR(W.grad().at({1, 1}), 6.0, 1e-12);
    CHECK_NEAR(b.grad().at({0, 0}), 2.0, 1e-12);
    CHECK_NEAR(b.grad().at({0, 1}), 2.0, 1e-12);
}

// ── mean ─────────────────────────────────────────────────────────────────────

inline void test_mean_forward_1d(Backend& be) {
    std::cout << "[OpsForward 26] mean: [1,2,3,4] == 2.5\n";
    Tensor x = Tensor::from_data<double>({1.0, 2.0, 3.0, 4.0}, {4}, be);
    Tensor y = x.mean();
    CHECK(y.shape() == (std::vector<std::size_t>{1}));
    CHECK_NEAR(y.at({0}), 2.5, 1e-12);
}

inline void test_mean_forward_2d(Backend& be) {
    std::cout << "[OpsForward 27] mean: [[1,2],[3,4]] == 2.5\n";
    Tensor x = Tensor::from_data<double>({1.0, 2.0, 3.0, 4.0}, {2, 2}, be);
    Tensor y = x.mean();
    CHECK_NEAR(y.at({0}), 2.5, 1e-12);
}

inline void test_mean_forward_scalar(Backend& be) {
    std::cout << "[OpsForward 28] mean: single-element tensor == that element\n";
    Tensor x = Tensor::from_data<double>({7.0}, {1}, be);
    CHECK_NEAR(x.mean().at({0}), 7.0, 1e-12);
}

inline void test_mean_forward_zeros(Backend& be) {
    std::cout << "[OpsForward 29] mean: all-zeros tensor == 0\n";
    Tensor x = Tensor::zeros({3, 3}, be);
    CHECK_NEAR(x.mean().at({0}), 0.0, 1e-12);
}

inline void test_mean_forward_noncontiguous(Backend& be) {
    std::cout << "[OpsForward 30] mean: transposed tensor == mean of original\n";
    Tensor x  = Tensor::from_data<double>({1.0, 2.0, 3.0, 4.0}, {2, 2}, be);
    Tensor xt = x.transpose(0, 1);
    CHECK(!xt.is_contiguous());
    CHECK_NEAR(xt.mean().at({0}), 2.5, 1e-12);
}

// ── run all ──────────────────────────────────────────────────────────────────

inline void run_shared_ops_forward(Backend& be) {
    test_add_forward_values(be);
    test_add_output_not_leaf(be);
    test_add_no_grad_when_inputs_detached(be);
    test_sum_forward_value(be);
    test_mul_forward_values(be);
    test_neg_forward(be);
    test_neg_no_grad_leaf(be);
    test_sub_forward(be);
    test_div_forward(be);
    test_div_ieee754_edge_cases(be);
    test_exp_forward(be);
    test_exp_ieee754_edge_cases(be);
    test_log_forward(be);
    test_log_ieee754_edge_cases(be);
    test_sqrt_forward(be);
    test_sqrt_ieee754_edge_cases(be);
    test_relu_forward(be);
    test_math_ops_graph_properties(be);
    test_math_ops_no_grad_leaf(be);
    test_matmul_2d_forward(be);
    test_matmul_identity_op(be);
    test_matmul_inner_mismatch_throws(be);
    test_matmul_batched_forward(be);
    test_matmul_batch_broadcast(be);
    test_matmul_linear_chain(be);
    test_mean_forward_1d(be);
    test_mean_forward_2d(be);
    test_mean_forward_scalar(be);
    test_mean_forward_zeros(be);
    test_mean_forward_noncontiguous(be);
}

} // namespace otter::test::shared
