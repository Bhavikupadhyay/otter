#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "../utils/test_utils.h"
#include "otter/tensor.h"
#include "otter/autograd/no_grad_guard.h"
#include "otter/kernel/backend.h"

namespace otter::test::shared {

// ── add + sum backward ────────────────────────────────────────────────────────

inline void test_add_backward_gradients(Backend& be) {
    std::cout << "[OpsBackward 1] add backward: grad_a == grad_b == ones\n";
    // loss = sum(a+b); d loss/d a_i = 1, d loss/d b_i = 1 for all i
    Tensor a = Tensor::from_data<double>({1, 2, 3}, {3}, be, /*requires_grad=*/true);
    Tensor b = Tensor::from_data<double>({4, 5, 6}, {3}, be, /*requires_grad=*/true);
    Tensor loss = a.add(b).sum();
    loss.backward();
    for (std::size_t i = 0; i < 3; ++i) {
        CHECK_NEAR(a.grad().at({i}), 1.0, 1e-12);
        CHECK_NEAR(b.grad().at({i}), 1.0, 1e-12);
    }
}

inline void test_sum_backward_fans_scalar(Backend& be) {
    std::cout << "[OpsBackward 2] sum backward: scalar grad fans out to all input elements\n";
    Tensor a = Tensor::from_data<double>({2, 4, 6, 8}, {2, 2}, be, /*requires_grad=*/true);
    a.sum().backward();
    // d(sum(a))/da_ij = 1 for all i,j
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 2; ++c)
            CHECK_NEAR(a.grad().at({r, c}), 1.0, 1e-12);
}

inline void test_add_chain_backward(Backend& be) {
    std::cout << "[OpsBackward 3] chain: (a + b + c).sum() — all three grads == 1\n";
    Tensor a = Tensor::from_data<double>({1, 2}, {2}, be, /*requires_grad=*/true);
    Tensor b = Tensor::from_data<double>({3, 4}, {2}, be, /*requires_grad=*/true);
    Tensor c = Tensor::from_data<double>({5, 6}, {2}, be, /*requires_grad=*/true);
    a.add(b).add(c).sum().backward();
    for (std::size_t i = 0; i < 2; ++i) {
        CHECK_NEAR(a.grad().at({i}), 1.0, 1e-12);
        CHECK_NEAR(b.grad().at({i}), 1.0, 1e-12);
        CHECK_NEAR(c.grad().at({i}), 1.0, 1e-12);
    }
}

// ── zero_grad ─────────────────────────────────────────────────────────────────

inline void test_zero_grad_clears_after_backward(Backend& be) {
    std::cout << "[OpsBackward 4] zero_grad: clears accumulated gradient\n";
    Tensor a = Tensor::from_data<double>({1, 2, 3}, {3}, be, /*requires_grad=*/true);
    a.add(Tensor::from_data<double>({4, 5, 6}, {3}, be)).sum().backward();
    CHECK(a.grad().defined());
    a.zero_grad();
    CHECK(!a.grad().defined());
}

// ── retain_graph ──────────────────────────────────────────────────────────────

inline void test_retain_graph_accumulates(Backend& be) {
    std::cout << "[OpsBackward 5] retain_graph: two backward passes accumulate gradients\n";
    Tensor a = Tensor::from_data<double>({1, 2}, {2}, be, /*requires_grad=*/true);
    Tensor b = Tensor::from_data<double>({3, 4}, {2}, be);
    Tensor loss = a.add(b).sum();
    loss.backward(/*retain_graph=*/true);
    // After 1st backward: grad_a = [1, 1]
    CHECK_NEAR(a.grad().at({0}), 1.0, 1e-12);
    CHECK_NEAR(a.grad().at({1}), 1.0, 1e-12);
    loss.backward(/*retain_graph=*/false);
    // After 2nd backward: grad_a = [2, 2] (accumulated)
    CHECK_NEAR(a.grad().at({0}), 2.0, 1e-12);
    CHECK_NEAR(a.grad().at({1}), 2.0, 1e-12);
}

inline void test_second_backward_after_cleanup_throws(Backend& be) {
    std::cout << "[OpsBackward 6] backward twice without retain_graph throws\n";
    Tensor a = Tensor::from_data<double>({1, 2}, {2}, be, /*requires_grad=*/true);
    Tensor loss = a.add(Tensor::from_data<double>({3, 4}, {2}, be)).sum();
    loss.backward();
    bool threw = false;
    try { loss.backward(); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

// ── NoGradGuard ───────────────────────────────────────────────────────────────

inline void test_no_grad_guard_skips_graph(Backend& be) {
    std::cout << "[OpsBackward 7] NoGradGuard: output has no grad graph\n";
    Tensor a = Tensor::from_data<double>({1, 2}, {2}, be, /*requires_grad=*/true);
    Tensor b = Tensor::from_data<double>({3, 4}, {2}, be);
    Tensor c;
    {
        NoGradGuard ng;
        c = a.add(b);
    }
    CHECK(!c.requires_grad());
    CHECK(c.is_leaf());
}

// ── detach ────────────────────────────────────────────────────────────────────

inline void test_detach_severs_grad_flow(Backend& be) {
    std::cout << "[OpsBackward 8] detach: gradient does not flow through detached tensor\n";
    Tensor a = Tensor::from_data<double>({1, 2}, {2}, be, /*requires_grad=*/true);
    Tensor b = Tensor::from_data<double>({3, 4}, {2}, be, /*requires_grad=*/true);
    Tensor loss = a.add(b.detach()).sum();
    loss.backward();
    CHECK(a.grad().defined());
    CHECK_NEAR(a.grad().at({0}), 1.0, 1e-12);
    CHECK_NEAR(a.grad().at({1}), 1.0, 1e-12);
    CHECK(!b.grad().defined());
}

// ── memory ────────────────────────────────────────────────────────────────────

inline void test_autograd_memory_clean(Backend& be) {
    std::cout << "[OpsBackward 9] memory: bytes_allocated == 0 after all tensors destruct\n";
    {
        Tensor a = Tensor::from_data<double>({1, 2, 3}, {3}, be, /*requires_grad=*/true);
        Tensor b = Tensor::from_data<double>({4, 5, 6}, {3}, be, /*requires_grad=*/true);
        a.add(b).sum().backward();
    }
    CHECK(be.memory_manager()->bytes_allocated() == 0);
}

// ── mul backward ─────────────────────────────────────────────────────────────

inline void test_mul_backward_gradients(Backend& be) {
    std::cout << "[OpsBackward 10] mul backward: grad_a = b, grad_b = a (product rule)\n";
    // loss = sum(a * b); d loss/d a_i = b_i, d loss/d b_i = a_i
    Tensor a = Tensor::from_data<double>({2, 3, 4}, {3}, be, /*requires_grad=*/true);
    Tensor b = Tensor::from_data<double>({5, 6, 7}, {3}, be, /*requires_grad=*/true);
    a.mul(b).sum().backward();
    CHECK_NEAR(a.grad().at({0}), 5.0, 1e-12);
    CHECK_NEAR(a.grad().at({1}), 6.0, 1e-12);
    CHECK_NEAR(a.grad().at({2}), 7.0, 1e-12);
    CHECK_NEAR(b.grad().at({0}), 2.0, 1e-12);
    CHECK_NEAR(b.grad().at({1}), 3.0, 1e-12);
    CHECK_NEAR(b.grad().at({2}), 4.0, 1e-12);
}

inline void test_add_mul_chain_backward(Backend& be) {
    std::cout << "[OpsBackward 11] chain: ((a + b) * c).sum() — product + add rule\n";
    // loss = sum((a+b)*c), grad_a = c, grad_b = c, grad_c = (a+b)
    Tensor a = Tensor::from_data<double>({1, 2}, {2}, be, /*requires_grad=*/true);
    Tensor b = Tensor::from_data<double>({3, 4}, {2}, be, /*requires_grad=*/true);
    Tensor c = Tensor::from_data<double>({5, 6}, {2}, be, /*requires_grad=*/true);
    a.add(b).mul(c).sum().backward();
    CHECK_NEAR(a.grad().at({0}), 5.0, 1e-12);
    CHECK_NEAR(a.grad().at({1}), 6.0, 1e-12);
    CHECK_NEAR(b.grad().at({0}), 5.0, 1e-12);
    CHECK_NEAR(b.grad().at({1}), 6.0, 1e-12);
    CHECK_NEAR(c.grad().at({0}), 4.0, 1e-12);
    CHECK_NEAR(c.grad().at({1}), 6.0, 1e-12);
}

inline void test_accumulate_grad_fresh_after_zero_grad(Backend& be) {
    std::cout << "[OpsBackward 12] zero_grad then backward: grad accumulates fresh, not doubled\n";
    Tensor a = Tensor::from_data<double>({1, 2, 3}, {3}, be, /*requires_grad=*/true);
    a.sum().backward();
    CHECK_NEAR(a.grad().at({0}), 1.0, 1e-12);
    a.zero_grad();
    CHECK(!a.grad().defined());
    a.sum().backward();
    // Should be [1, 1, 1] again, not [2, 2, 2]
    CHECK_NEAR(a.grad().at({0}), 1.0, 1e-12);
    CHECK_NEAR(a.grad().at({1}), 1.0, 1e-12);
    CHECK_NEAR(a.grad().at({2}), 1.0, 1e-12);
}

inline void test_backward_releases_saved_inputs(Backend& be) {
    std::cout << "[OpsBackward 13] backward(retain_graph=false): saved_inputs cleared, memory freed\n";
    {
        Tensor a = Tensor::from_data<double>({1, 2, 3}, {3}, be, /*requires_grad=*/true);
        Tensor b = Tensor::from_data<double>({4, 5, 6}, {3}, be, /*requires_grad=*/true);
        Tensor c = a.add(b);
        Tensor d = c.mul(b);
        d.sum().backward();
    }
    CHECK(be.memory_manager()->bytes_allocated() == 0);
}

// ── neg backward ─────────────────────────────────────────────────────────────

inline void test_neg_backward(Backend& be) {
    std::cout << "[OpsBackward 14] neg: backward gradient is -1 everywhere\n";
    // loss = sum(-a), d(loss)/da = -1 for all elements
    Tensor a = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be, true);
    a.neg().sum().backward();
    CHECK(a.grad().defined());
    CHECK_NEAR(a.grad().at({0}), -1.0, 1e-9);
    CHECK_NEAR(a.grad().at({1}), -1.0, 1e-9);
    CHECK_NEAR(a.grad().at({2}), -1.0, 1e-9);
}

// ── sub backward ─────────────────────────────────────────────────────────────

inline void test_sub_backward(Backend& be) {
    std::cout << "[OpsBackward 15] sub: grad_a = grad_out, grad_b = -grad_out\n";
    // loss = sum(a - b), grad_a[i] = 1, grad_b[i] = -1
    Tensor a = Tensor::from_data<double>({3.0, 1.0}, {2}, be, true);
    Tensor b = Tensor::from_data<double>({1.0, 2.0}, {2}, be, true);
    a.sub(b).sum().backward();
    CHECK_NEAR(a.grad().at({0}),  1.0, 1e-9);
    CHECK_NEAR(a.grad().at({1}),  1.0, 1e-9);
    CHECK_NEAR(b.grad().at({0}), -1.0, 1e-9);
    CHECK_NEAR(b.grad().at({1}), -1.0, 1e-9);
}

inline void test_sub_broadcast_backward(Backend& be) {
    std::cout << "[OpsBackward 16] sub: broadcast backward reduces grad to original shape\n";
    // a: {1,3}, b: {2,3} → grad_a sums over batch rows: [2,2,2]
    Tensor a = Tensor::from_data<double>({1.0, 2.0, 3.0}, {1, 3}, be, true);
    Tensor b = Tensor::from_data<double>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {2, 3}, be, true);
    a.sub(b).sum().backward();
    CHECK_NEAR(a.grad().at({0, 0}), 2.0, 1e-9);
    CHECK_NEAR(a.grad().at({0, 1}), 2.0, 1e-9);
    CHECK_NEAR(a.grad().at({0, 2}), 2.0, 1e-9);
}

// ── div backward ─────────────────────────────────────────────────────────────

inline void test_div_backward(Backend& be) {
    std::cout << "[OpsBackward 17] div: grad_a = grad/b, grad_b = -grad*a/(b*b)\n";
    // loss = sum(a / b), a=[6], b=[2]
    // grad_a = 1/b = 0.5, grad_b = -a/(b*b) = -6/4 = -1.5
    Tensor a = Tensor::from_data<double>({6.0}, {1}, be, true);
    Tensor b = Tensor::from_data<double>({2.0}, {1}, be, true);
    a.div(b).sum().backward();
    CHECK_NEAR(a.grad().at({0}),  0.5,  1e-9);  // 1/2
    CHECK_NEAR(b.grad().at({0}), -1.5,  1e-9);  // -6/(2*2)
}

inline void test_div_backward_numerical(Backend& be) {
    std::cout << "[OpsBackward 18] div: numerical gradient check\n";
    auto fwd = [&](Tensor a_p) {
        Tensor bv = Tensor::from_data<double>({3.0, 4.0}, {2}, be);
        return a_p.div(bv).sum();
    };

    Tensor a = Tensor::from_data<double>({6.0, 8.0}, {2}, be, true);
    Tensor b = Tensor::from_data<double>({3.0, 4.0}, {2}, be, true);
    a.div(b).sum().backward();

    constexpr double eps = 1e-5;
    for (std::size_t i = 0; i < 2; ++i) {
        std::vector<double> data = {6.0, 8.0};
        data[i] += eps;
        double fp = fwd(Tensor::from_data<double>(data, {2}, be)).at({0});
        data[i] -= 2.0 * eps;
        double fm = fwd(Tensor::from_data<double>(data, {2}, be)).at({0});
        double num = (fp - fm) / (2.0 * eps);
        CHECK_NEAR(a.grad().at({i}), num, 1e-4);
    }
}

// ── exp backward ─────────────────────────────────────────────────────────────

inline void test_exp_backward(Backend& be) {
    std::cout << "[OpsBackward 19] exp: backward grad = grad_out * exp(x)\n";
    // loss = sum(exp(a)), d(loss)/da[i] = exp(a[i])
    Tensor a = Tensor::from_data<double>({0.0, 1.0}, {2}, be, true);
    a.exp().sum().backward();
    CHECK_NEAR(a.grad().at({0}), std::exp(0.0), 1e-9);  // exp(0) = 1
    CHECK_NEAR(a.grad().at({1}), std::exp(1.0), 1e-9);  // exp(1) ≈ 2.71828
}

inline void test_exp_backward_numerical(Backend& be) {
    std::cout << "[OpsBackward 20] exp: numerical gradient check\n";
    Tensor a = Tensor::from_data<double>({0.5, -0.5}, {2}, be, true);
    a.exp().sum().backward();
    constexpr double eps = 1e-5;
    for (std::size_t i = 0; i < 2; ++i) {
        std::vector<double> data = {0.5, -0.5};
        data[i] += eps;
        double fp = Tensor::from_data<double>(data, {2}, be).exp().sum().at({0});
        data[i] -= 2.0 * eps;
        double fm = Tensor::from_data<double>(data, {2}, be).exp().sum().at({0});
        CHECK_NEAR(a.grad().at({i}), (fp - fm) / (2.0 * eps), 1e-4);
    }
}

// ── log backward ─────────────────────────────────────────────────────────────

inline void test_log_backward(Backend& be) {
    std::cout << "[OpsBackward 21] log: backward grad = grad_out / x\n";
    // loss = sum(log(a)), d(loss)/da[i] = 1/a[i]
    Tensor a = Tensor::from_data<double>({2.0, 4.0}, {2}, be, true);
    a.log().sum().backward();
    CHECK_NEAR(a.grad().at({0}), 0.5,  1e-9);  // 1/2
    CHECK_NEAR(a.grad().at({1}), 0.25, 1e-9);  // 1/4
}

inline void test_log_backward_numerical(Backend& be) {
    std::cout << "[OpsBackward 22] log: numerical gradient check\n";
    Tensor a = Tensor::from_data<double>({1.0, 2.0}, {2}, be, true);
    a.log().sum().backward();
    constexpr double eps = 1e-5;
    for (std::size_t i = 0; i < 2; ++i) {
        std::vector<double> data = {1.0, 2.0};
        data[i] += eps;
        double fp = Tensor::from_data<double>(data, {2}, be).log().sum().at({0});
        data[i] -= 2.0 * eps;
        double fm = Tensor::from_data<double>(data, {2}, be).log().sum().at({0});
        CHECK_NEAR(a.grad().at({i}), (fp - fm) / (2.0 * eps), 1e-4);
    }
}

// ── sqrt backward ────────────────────────────────────────────────────────────

inline void test_sqrt_backward(Backend& be) {
    std::cout << "[OpsBackward 23] sqrt: backward grad = grad_out / (2 * sqrt(x))\n";
    // loss = sum(sqrt(a)), d(loss)/da[i] = 1/(2*sqrt(a[i]))
    // a=[4, 9]: grads = [0.25, 1/6]
    Tensor a = Tensor::from_data<double>({4.0, 9.0}, {2}, be, true);
    a.sqrt().sum().backward();
    CHECK_NEAR(a.grad().at({0}), 0.25,    1e-9);  // 1/(2*2) = 0.25
    CHECK_NEAR(a.grad().at({1}), 1.0/6.0, 1e-9);  // 1/(2*3)
}

inline void test_sqrt_backward_zero_input(Backend& be) {
    std::cout << "[OpsBackward 24] sqrt: grad at x=0 is +inf (IEEE 754)\n";
    Tensor a = Tensor::from_data<double>({0.0}, {1}, be, true);
    a.sqrt().sum().backward();
    // grad = 1 / (2 * sqrt(0)) = +inf
    CHECK(std::isinf(a.grad().at({0})) && a.grad().at({0}) > 0.0);
}

inline void test_sqrt_backward_numerical(Backend& be) {
    std::cout << "[OpsBackward 25] sqrt: numerical gradient check\n";
    Tensor a = Tensor::from_data<double>({1.0, 4.0}, {2}, be, true);
    a.sqrt().sum().backward();
    constexpr double eps = 1e-5;
    for (std::size_t i = 0; i < 2; ++i) {
        std::vector<double> data = {1.0, 4.0};
        data[i] += eps;
        double fp = Tensor::from_data<double>(data, {2}, be).sqrt().sum().at({0});
        data[i] -= 2.0 * eps;
        double fm = Tensor::from_data<double>(data, {2}, be).sqrt().sum().at({0});
        CHECK_NEAR(a.grad().at({i}), (fp - fm) / (2.0 * eps), 1e-4);
    }
}

// ── relu backward ────────────────────────────────────────────────────────────

inline void test_relu_backward_values(Backend& be) {
    std::cout << "[OpsBackward 26] relu: grad passes where x>0, zero where x<=0\n";
    // x=[3, -5, 0], grad_out=[1,1,1] (from sum) → grad_x=[1,0,0]
    Tensor a = Tensor::from_data<double>({3.0, -5.0, 0.0}, {3}, be, true);
    a.relu().sum().backward();
    CHECK_NEAR(a.grad().at({0}), 1.0, 1e-9);
    CHECK_NEAR(a.grad().at({1}), 0.0, 1e-9);
    CHECK_NEAR(a.grad().at({2}), 0.0, 1e-9);
}

inline void test_relu_backward_numerical(Backend& be) {
    std::cout << "[OpsBackward 27] relu: numerical gradient check (positive region)\n";
    Tensor a = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be, true);
    a.relu().sum().backward();
    constexpr double eps = 1e-5;
    for (std::size_t i = 0; i < 3; ++i) {
        std::vector<double> data = {1.0, 2.0, 3.0};
        data[i] += eps;
        double fp = Tensor::from_data<double>(data, {3}, be).relu().sum().at({0});
        data[i] -= 2.0 * eps;
        double fm = Tensor::from_data<double>(data, {3}, be).relu().sum().at({0});
        CHECK_NEAR(a.grad().at({i}), (fp - fm) / (2.0 * eps), 1e-4);
    }
}

// ── composition ──────────────────────────────────────────────────────────────

inline void test_math_ops_composition_backward(Backend& be) {
    std::cout << "[OpsBackward 28] composition: log(exp(x)) backward = 1 everywhere\n";
    Tensor a = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be, true);
    a.exp().log().sum().backward();
    CHECK_NEAR(a.grad().at({0}), 1.0, 1e-6);
    CHECK_NEAR(a.grad().at({1}), 1.0, 1e-6);
    CHECK_NEAR(a.grad().at({2}), 1.0, 1e-6);
}

inline void test_math_ops_retain_graph(Backend& be) {
    std::cout << "[OpsBackward 29] retain_graph: two backward passes accumulate gradients\n";
    Tensor a = Tensor::from_data<double>({2.0, 3.0}, {2}, be, true);
    Tensor loss = a.neg().sum();
    loss.backward(true);   // retain graph
    loss.backward(true);   // second pass — should accumulate
    // Each pass contributes -1 per element → total = -2
    CHECK_NEAR(a.grad().at({0}), -2.0, 1e-9);
    CHECK_NEAR(a.grad().at({1}), -2.0, 1e-9);
}

// ── matmul backward ──────────────────────────────────────────────────────────

inline void test_matmul_backward_2d(Backend& be) {
    std::cout << "[OpsBackward 30] matmul backward 2x3 @ 3x2: grad_A = grad@B^T, grad_B = A^T@grad\n";
    // A = [[1,2,3],[4,5,6]], B = [[1,0],[0,1],[1,1]]
    // grad_A = [[1,1],[1,1]] @ B^T → rows: [1,1,2]
    // grad_B = A^T @ [[1,1],[1,1]] → [[5,5],[7,7],[9,9]]
    Tensor a = Tensor::from_data<double>({1,2,3, 4,5,6},  {2, 3}, be, /*requires_grad=*/true);
    Tensor b = Tensor::from_data<double>({1,0, 0,1, 1,1}, {3, 2}, be, /*requires_grad=*/true);
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

// ── mean backward ────────────────────────────────────────────────────────────

inline void test_mean_backward_analytical(Backend& be) {
    std::cout << "[OpsBackward 31] mean backward: grad_input[i] == 1/n for all i\n";
    Tensor x = Tensor::from_data<double>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                                         {2, 3}, be, /*requires_grad=*/true);
    x.mean().backward();
    auto g = x.grad().to_vector<double>();
    CHECK(g.size() == 6);
    for (double gi : g)
        CHECK_NEAR(gi, 1.0 / 6.0, 1e-12);
}

inline void test_mean_backward_seed_scaling(Backend& be) {
    std::cout << "[OpsBackward 32] mean backward: grad scales with incoming seed\n";
    // seed = 3.0 → grad_input[i] = 3.0/n
    Tensor x    = Tensor::from_data<double>({1.0, 2.0, 3.0, 4.0}, {4}, be, /*requires_grad=*/true);
    Tensor seed = Tensor::from_data<double>({3.0}, {1}, be);
    x.mean().backward(seed);
    for (double gi : x.grad().to_vector<double>())
        CHECK_NEAR(gi, 3.0 / 4.0, 1e-12);
}

inline void test_mean_backward_numerical(Backend& be) {
    std::cout << "[OpsBackward 33] mean backward: numerical gradient check\n";
    const std::vector<double> data = {1.0, -2.0, 0.5, 3.0};
    Tensor x = Tensor::from_data<double>(data, {4}, be, /*requires_grad=*/true);
    x.mean().backward();
    auto analytic = x.grad().to_vector<double>();
    const double eps = 1e-5;
    for (std::size_t i = 0; i < data.size(); ++i) {
        std::vector<double> dp = data, dm = data;
        dp[i] += eps; dm[i] -= eps;
        double num = (Tensor::from_data<double>(dp, {4}, be).mean().at({0})
                    - Tensor::from_data<double>(dm, {4}, be).mean().at({0}))
                   / (2.0 * eps);
        CHECK_NEAR(analytic[i], num, 1e-4);
    }
}

inline void test_mean_graph_computed(Backend& be) {
    std::cout << "[OpsBackward 34] mean graph: output is not a leaf when input requires_grad\n";
    Tensor x = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be, /*requires_grad=*/true);
    Tensor y = x.mean();
    CHECK(!y.is_leaf());
    CHECK(y.requires_grad());
}

inline void test_mean_graph_no_grad(Backend& be) {
    std::cout << "[OpsBackward 35] mean graph: output is leaf when input has no grad\n";
    Tensor x = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be);
    Tensor y = x.mean();
    CHECK(y.is_leaf());
    CHECK(!y.requires_grad());
}

inline void test_mean_no_second_backward(Backend& be) {
    std::cout << "[OpsBackward 36] mean: second backward throws after graph cleared\n";
    Tensor x    = Tensor::from_data<double>({1.0, 2.0}, {2}, be, /*requires_grad=*/true);
    Tensor loss = x.mean();
    loss.backward();
    bool threw = false;
    try { loss.backward(); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

// ── ops memory clean ─────────────────────────────────────────────────────────

inline void test_math_ops_memory_clean(Backend& be) {
    std::cout << "[OpsBackward 37] memory: bytes_allocated == 0 after math ops and backward\n";
    {
        Tensor a = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be, true);
        Tensor b = Tensor::from_data<double>({4.0, 5.0, 6.0}, {3}, be, true);
        a.neg().sum().backward();    a.zero_grad();
        a.sub(b).sum().backward();   a.zero_grad(); b.zero_grad();
        a.div(b).sum().backward();   a.zero_grad(); b.zero_grad();
        a.exp().sum().backward();    a.zero_grad();
        a.log().sum().backward();    a.zero_grad();
        Tensor pos = Tensor::from_data<double>({1.0, 4.0, 9.0}, {3}, be, true);
        pos.sqrt().sum().backward(); pos.zero_grad();
        pos.relu().sum().backward(); pos.zero_grad();
    }
    CHECK(be.memory_manager()->bytes_allocated() == 0);
}

// ── run all ──────────────────────────────────────────────────────────────────

inline void run_shared_ops_backward(Backend& be) {
    test_add_backward_gradients(be);
    test_sum_backward_fans_scalar(be);
    test_add_chain_backward(be);
    test_zero_grad_clears_after_backward(be);
    test_retain_graph_accumulates(be);
    test_second_backward_after_cleanup_throws(be);
    test_no_grad_guard_skips_graph(be);
    test_detach_severs_grad_flow(be);
    test_autograd_memory_clean(be);
    test_mul_backward_gradients(be);
    test_add_mul_chain_backward(be);
    test_accumulate_grad_fresh_after_zero_grad(be);
    test_backward_releases_saved_inputs(be);
    test_neg_backward(be);
    test_sub_backward(be);
    test_sub_broadcast_backward(be);
    test_div_backward(be);
    test_div_backward_numerical(be);
    test_exp_backward(be);
    test_exp_backward_numerical(be);
    test_log_backward(be);
    test_log_backward_numerical(be);
    test_sqrt_backward(be);
    test_sqrt_backward_zero_input(be);
    test_sqrt_backward_numerical(be);
    test_relu_backward_values(be);
    test_relu_backward_numerical(be);
    test_math_ops_composition_backward(be);
    test_math_ops_retain_graph(be);
    test_matmul_backward_2d(be);
    test_mean_backward_analytical(be);
    test_mean_backward_seed_scaling(be);
    test_mean_backward_numerical(be);
    test_mean_graph_computed(be);
    test_mean_graph_no_grad(be);
    test_mean_no_second_backward(be);
    test_math_ops_memory_clean(be);
}

} // namespace otter::test::shared
