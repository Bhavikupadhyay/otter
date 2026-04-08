#include "test_utils.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "otter/tensor.h"
#include "otter/backends/cpu.h"

// Tests for elementwise math operations: neg, sub, div, exp, log, sqrt, relu.
// Covers: forward values, IEEE 754 edge cases (nan/inf), backward gradients,
// numerical gradient checks, graph properties, and memory cleanliness.

namespace otter::test {

// ── neg ───────────────────────────────────────────────────────────────────────

void test_neg_forward() {
    std::cout << "[MathOps 1] neg: forward values and shape\n";
    Backend& be = cpu_backend();
    // [-1.0, 2.0, -3.0] → [1.0, -2.0, 3.0]
    Tensor a = Tensor::from_data<double>({-1.0, 2.0, -3.0}, {3}, be);
    Tensor b = a.neg();
    CHECK(b.shape() == (std::vector<std::size_t>{3}));
    CHECK_NEAR(b.at({0}), 1.0, 1e-12);   // -(-1) = 1
    CHECK_NEAR(b.at({1}), -2.0, 1e-12);  // -(2) = -2
    CHECK_NEAR(b.at({2}), 3.0, 1e-12);   // -(-3) = 3
}

void test_neg_backward() {
    std::cout << "[MathOps 2] neg: backward gradient is -1 everywhere\n";
    Backend& be = cpu_backend();
    // loss = sum(-a),  d(loss)/da = -1 for all elements
    Tensor a = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be, true);
    Tensor loss = a.neg().sum();
    loss.backward();
    CHECK(a.grad().defined());
    CHECK_NEAR(a.grad().at({0}), -1.0, 1e-9);  // d(-a0)/da0 = -1
    CHECK_NEAR(a.grad().at({1}), -1.0, 1e-9);
    CHECK_NEAR(a.grad().at({2}), -1.0, 1e-9);
}

void test_neg_no_grad_leaf() {
    std::cout << "[MathOps 3] neg: no grad graph when input has requires_grad=false\n";
    Backend& be = cpu_backend();
    Tensor a = Tensor::from_data<double>({1.0, 2.0}, {2}, be);  // requires_grad=false
    Tensor b = a.neg();
    // No input requires grad → execute() does not wire a grad graph → b is a plain leaf.
    CHECK(!a.requires_grad());
    CHECK(b.is_leaf());
}

// ── sub ───────────────────────────────────────────────────────────────────────

void test_sub_forward() {
    std::cout << "[MathOps 4] sub: forward values\n";
    Backend& be = cpu_backend();
    // [5, 3, 1] - [1, 2, 3] = [4, 1, -2]
    Tensor a = Tensor::from_data<double>({5.0, 3.0, 1.0}, {3}, be);
    Tensor b = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be);
    Tensor c = a.sub(b);
    CHECK(c.shape() == (std::vector<std::size_t>{3}));
    CHECK_NEAR(c.at({0}), 4.0, 1e-12);
    CHECK_NEAR(c.at({1}), 1.0, 1e-12);
    CHECK_NEAR(c.at({2}), -2.0, 1e-12);
}

void test_sub_backward() {
    std::cout << "[MathOps 5] sub: grad_a = grad_out, grad_b = -grad_out\n";
    Backend& be = cpu_backend();
    // loss = sum(a - b), grad_a[i] = 1, grad_b[i] = -1
    Tensor a = Tensor::from_data<double>({3.0, 1.0}, {2}, be, true);
    Tensor b = Tensor::from_data<double>({1.0, 2.0}, {2}, be, true);
    Tensor loss = a.sub(b).sum();
    loss.backward();
    CHECK(a.grad().defined());
    CHECK(b.grad().defined());
    CHECK_NEAR(a.grad().at({0}), 1.0, 1e-9);   // d(a-b)/da = 1
    CHECK_NEAR(a.grad().at({1}), 1.0, 1e-9);
    CHECK_NEAR(b.grad().at({0}), -1.0, 1e-9);  // d(a-b)/db = -1
    CHECK_NEAR(b.grad().at({1}), -1.0, 1e-9);
}

void test_sub_broadcast_backward() {
    std::cout << "[MathOps 6] sub: broadcast backward reduces grad to original shape\n";
    Backend& be = cpu_backend();
    // a: {1,3},  b: {2,3}  → out: {2,3}
    // grad_a must be reduced from {2,3} back to {1,3}
    Tensor a = Tensor::from_data<double>({1.0, 2.0, 3.0}, {1, 3}, be, true);
    Tensor b = Tensor::from_data<double>({0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0}, {2, 3}, be, true);
    Tensor loss = a.sub(b).sum();
    loss.backward();
    // grad_a[0,j] = sum over batch rows = 1+1 = 2
    CHECK_NEAR(a.grad().at({0, 0}), 2.0, 1e-9);
    CHECK_NEAR(a.grad().at({0, 1}), 2.0, 1e-9);
    CHECK_NEAR(a.grad().at({0, 2}), 2.0, 1e-9);
}

// ── div ───────────────────────────────────────────────────────────────────────

void test_div_forward() {
    std::cout << "[MathOps 7] div: forward values\n";
    Backend& be = cpu_backend();
    // [6, 8, 9] / [2, 4, 3] = [3, 2, 3]
    Tensor a = Tensor::from_data<double>({6.0, 8.0, 9.0}, {3}, be);
    Tensor b = Tensor::from_data<double>({2.0, 4.0, 3.0}, {3}, be);
    Tensor c = a.div(b);
    CHECK_NEAR(c.at({0}), 3.0, 1e-12);
    CHECK_NEAR(c.at({1}), 2.0, 1e-12);
    CHECK_NEAR(c.at({2}), 3.0, 1e-12);
}

void test_div_ieee754_edge_cases() {
    std::cout << "[MathOps 8] div: IEEE 754 edge cases — div by zero\n";
    Backend& be = cpu_backend();
    // 1.0 / 0.0 → +inf
    Tensor a1 = Tensor::from_data<double>({ 1.0}, {1}, be);
    Tensor b0 = Tensor::from_data<double>({ 0.0}, {1}, be);
    Tensor r1 = a1.div(b0);
    CHECK(std::isinf(r1.at({0})) && r1.at({0}) > 0.0);  // +inf

    // -1.0 / 0.0 → -inf
    Tensor am = Tensor::from_data<double>({-1.0}, {1}, be);
    Tensor r2 = am.div(b0);
    CHECK(std::isinf(r2.at({0})) && r2.at({0}) < 0.0);  // -inf

    // 0.0 / 0.0 → nan
    Tensor a0 = Tensor::from_data<double>({ 0.0}, {1}, be);
    Tensor r3 = a0.div(b0);
    CHECK(std::isnan(r3.at({0})));  // nan
}

void test_div_backward() {
    std::cout << "[MathOps 9] div: backward grad_a = grad/b, grad_b = -grad*a/(b*b)\n";
    Backend& be = cpu_backend();
    // loss = sum(a / b), a=[6], b=[2]
    // grad_a = 1/b = 1/2 = 0.5
    // grad_b = -a/(b*b) = -6/4 = -1.5
    Tensor a = Tensor::from_data<double>({6.0}, {1}, be, true);
    Tensor b = Tensor::from_data<double>({2.0}, {1}, be, true);
    Tensor loss = a.div(b).sum();
    loss.backward();
    CHECK_NEAR(a.grad().at({0}), 0.5, 1e-9);    // 1/2
    CHECK_NEAR(b.grad().at({0}), -1.5, 1e-9);   // -6/(2*2)
}

void test_div_backward_numerical() {
    std::cout << "[MathOps 10] div: numerical gradient check\n";
    Backend& be = cpu_backend();
    auto fwd = [&](Tensor a_p) {
        Tensor b = Tensor::from_data<double>({3.0, 4.0}, {2}, be);
        return a_p.div(b).sum();
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

// ── exp ───────────────────────────────────────────────────────────────────────

void test_exp_forward() {
    std::cout << "[MathOps 11] exp: forward values\n";
    Backend& be = cpu_backend();
    // exp(0) = 1, exp(1) ≈ 2.71828
    Tensor a = Tensor::from_data<double>({0.0, 1.0}, {2}, be);
    Tensor b = a.exp();
    CHECK_NEAR(b.at({0}), 1.0,              1e-12);
    CHECK_NEAR(b.at({1}), std::exp(1.0),    1e-12);
}

void test_exp_ieee754_edge_cases() {
    std::cout << "[MathOps 12] exp: IEEE 754 edge cases — overflow and underflow\n";
    Backend& be = cpu_backend();
    // exp(1000) → +inf (overflow)
    Tensor hi = Tensor::from_data<double>({1000.0}, {1}, be);
    Tensor rhi = hi.exp();
    CHECK(std::isinf(rhi.at({0})) && rhi.at({0}) > 0.0);

    // exp(-1000) → 0.0 (underflow)
    Tensor lo = Tensor::from_data<double>({-1000.0}, {1}, be);
    Tensor rlo = lo.exp();
    CHECK_NEAR(rlo.at({0}), 0.0, 1e-300);
}

void test_exp_backward() {
    std::cout << "[MathOps 13] exp: backward grad = grad_out * exp(x)\n";
    Backend& be = cpu_backend();
    // loss = sum(exp(a)),  d(loss)/da[i] = exp(a[i])
    Tensor a = Tensor::from_data<double>({0.0, 1.0}, {2}, be, true);
    Tensor loss = a.exp().sum();
    loss.backward();
    CHECK_NEAR(a.grad().at({0}), std::exp(0.0), 1e-9);  // exp(0) = 1
    CHECK_NEAR(a.grad().at({1}), std::exp(1.0), 1e-9);  // exp(1) ≈ 2.71828
}

void test_exp_backward_numerical() {
    std::cout << "[MathOps 14] exp: numerical gradient check\n";
    Backend& be = cpu_backend();
    Tensor a = Tensor::from_data<double>({0.5, -0.5}, {2}, be, true);
    a.exp().sum().backward();

    constexpr double eps = 1e-5;
    for (std::size_t i = 0; i < 2; ++i) {
        std::vector<double> data = {0.5, -0.5};
        data[i] += eps;
        double fp = Tensor::from_data<double>(data, {2}, be).exp().sum().at({0});
        data[i] -= 2.0 * eps;
        double fm = Tensor::from_data<double>(data, {2}, be).exp().sum().at({0});
        double num = (fp - fm) / (2.0 * eps);
        CHECK_NEAR(a.grad().at({i}), num, 1e-4);
    }
}

// ── log ───────────────────────────────────────────────────────────────────────

void test_log_forward() {
    std::cout << "[MathOps 15] log: forward values\n";
    Backend& be = cpu_backend();
    // log(1) = 0, log(e) = 1
    Tensor a = Tensor::from_data<double>({1.0, std::exp(1.0)}, {2}, be);
    Tensor b = a.log();
    CHECK_NEAR(b.at({0}), 0.0, 1e-12);
    CHECK_NEAR(b.at({1}), 1.0, 1e-12);
}

void test_log_ieee754_edge_cases() {
    std::cout << "[MathOps 16] log: IEEE 754 edge cases — log(0) = -inf, log(-1) = nan\n";
    Backend& be = cpu_backend();
    // log(0.0) → -inf
    Tensor z = Tensor::from_data<double>({0.0}, {1}, be);
    Tensor rz = z.log();
    CHECK(std::isinf(rz.at({0})) && rz.at({0}) < 0.0);  // -inf

    // log(-1.0) → nan
    Tensor neg = Tensor::from_data<double>({-1.0}, {1}, be);
    Tensor rneg = neg.log();
    CHECK(std::isnan(rneg.at({0})));  // nan
}

void test_log_backward() {
    std::cout << "[MathOps 17] log: backward grad = grad_out / x\n";
    Backend& be = cpu_backend();
    // loss = sum(log(a)),  d(loss)/da[i] = 1/a[i]
    Tensor a = Tensor::from_data<double>({2.0, 4.0}, {2}, be, true);
    Tensor loss = a.log().sum();
    loss.backward();
    CHECK_NEAR(a.grad().at({0}), 0.5, 1e-9);   // 1/2
    CHECK_NEAR(a.grad().at({1}), 0.25, 1e-9);  // 1/4
}

void test_log_backward_numerical() {
    std::cout << "[MathOps 18] log: numerical gradient check\n";
    Backend& be = cpu_backend();
    Tensor a = Tensor::from_data<double>({1.0, 2.0}, {2}, be, true);
    a.log().sum().backward();

    constexpr double eps = 1e-5;
    for (std::size_t i = 0; i < 2; ++i) {
        std::vector<double> data = {1.0, 2.0};
        data[i] += eps;
        double fp = Tensor::from_data<double>(data, {2}, be).log().sum().at({0});
        data[i] -= 2.0 * eps;
        double fm = Tensor::from_data<double>(data, {2}, be).log().sum().at({0});
        double num = (fp - fm) / (2.0 * eps);
        CHECK_NEAR(a.grad().at({i}), num, 1e-4);
    }
}

// ── sqrt ──────────────────────────────────────────────────────────────────────

void test_sqrt_forward() {
    std::cout << "[MathOps 19] sqrt: forward values\n";
    Backend& be = cpu_backend();
    // sqrt(4) = 2, sqrt(9) = 3
    Tensor a = Tensor::from_data<double>({4.0, 9.0}, {2}, be);
    Tensor b = a.sqrt();
    CHECK_NEAR(b.at({0}), 2.0, 1e-12);
    CHECK_NEAR(b.at({1}), 3.0, 1e-12);
}

void test_sqrt_ieee754_edge_cases() {
    std::cout << "[MathOps 20] sqrt: IEEE 754 edge cases — sqrt(-1) = nan, sqrt(0) = 0\n";
    Backend& be = cpu_backend();
    // sqrt(-1.0) → nan
    Tensor neg = Tensor::from_data<double>({-1.0}, {1}, be);
    Tensor rneg = neg.sqrt();
    CHECK(std::isnan(rneg.at({0})));  // nan

    // sqrt(0.0) → 0.0
    Tensor z = Tensor::from_data<double>({0.0}, {1}, be);
    Tensor rz = z.sqrt();
    CHECK_NEAR(rz.at({0}), 0.0, 1e-12);
}

void test_sqrt_backward() {
    std::cout << "[MathOps 21] sqrt: backward grad = grad_out / (2 * sqrt(x))\n";
    Backend& be = cpu_backend();
    // loss = sum(sqrt(a)),  d(loss)/da[i] = 1/(2*sqrt(a[i]))
    // a=[4, 9]: grads = [1/(2*2), 1/(2*3)] = [0.25, 1/6]
    Tensor a = Tensor::from_data<double>({4.0, 9.0}, {2}, be, true);
    Tensor loss = a.sqrt().sum();
    loss.backward();
    CHECK_NEAR(a.grad().at({0}), 0.25,        1e-9);  // 1/(2*2) = 0.25
    CHECK_NEAR(a.grad().at({1}), 1.0/6.0,     1e-9);  // 1/(2*3)
}

void test_sqrt_backward_zero_input() {
    std::cout << "[MathOps 22] sqrt: grad at x=0 is +inf (IEEE 754)\n";
    Backend& be = cpu_backend();
    Tensor a = Tensor::from_data<double>({0.0}, {1}, be, true);
    Tensor loss = a.sqrt().sum();
    loss.backward();
    // grad = 1 / (2 * sqrt(0)) = 1/0 = +inf
    CHECK(std::isinf(a.grad().at({0})) && a.grad().at({0}) > 0.0);
}

void test_sqrt_backward_numerical() {
    std::cout << "[MathOps 23] sqrt: numerical gradient check\n";
    Backend& be = cpu_backend();
    Tensor a = Tensor::from_data<double>({1.0, 4.0}, {2}, be, true);
    a.sqrt().sum().backward();

    constexpr double eps = 1e-5;
    for (std::size_t i = 0; i < 2; ++i) {
        std::vector<double> data = {1.0, 4.0};
        data[i] += eps;
        double fp = Tensor::from_data<double>(data, {2}, be).sqrt().sum().at({0});
        data[i] -= 2.0 * eps;
        double fm = Tensor::from_data<double>(data, {2}, be).sqrt().sum().at({0});
        double num = (fp - fm) / (2.0 * eps);
        CHECK_NEAR(a.grad().at({i}), num, 1e-4);
    }
}

// ── relu ──────────────────────────────────────────────────────────────────────

void test_relu_forward() {
    std::cout << "[MathOps 24] relu: forward — positive passes, negative zeros\n";
    Backend& be = cpu_backend();
    Tensor a = Tensor::from_data<double>({3.0, -5.0, 0.0}, {3}, be);
    Tensor b = a.relu();
    CHECK_NEAR(b.at({0}), 3.0, 1e-12);  // positive: pass through
    CHECK_NEAR(b.at({1}), 0.0, 1e-12);  // negative: zero
    CHECK_NEAR(b.at({2}), 0.0, 1e-12);  // zero: zero (right-hand derivative)
}

void test_relu_backward_values() {
    std::cout << "[MathOps 25] relu: backward — grad passes where x>0, zero where x<=0\n";
    Backend& be = cpu_backend();
    // x=[3, -5, 0], grad_out=[1,1,1] (from sum)
    // grad_x = [1*1, 1*0, 1*0] = [1, 0, 0]
    Tensor a = Tensor::from_data<double>({3.0, -5.0, 0.0}, {3}, be, true);
    Tensor loss = a.relu().sum();
    loss.backward();
    CHECK_NEAR(a.grad().at({0}), 1.0, 1e-9);  // x>0: gradient passes
    CHECK_NEAR(a.grad().at({1}), 0.0, 1e-9);  // x<0: gradient blocked
    CHECK_NEAR(a.grad().at({2}), 0.0, 1e-9);  // x=0: subgradient 0
}

void test_relu_backward_numerical() {
    std::cout << "[MathOps 26] relu: numerical gradient check (positive region)\n";
    Backend& be = cpu_backend();
    // Use x > 0 so numerical check is well-defined (no kink)
    Tensor a = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be, true);
    a.relu().sum().backward();

    constexpr double eps = 1e-5;
    for (std::size_t i = 0; i < 3; ++i) {
        std::vector<double> data = {1.0, 2.0, 3.0};
        data[i] += eps;
        double fp = Tensor::from_data<double>(data, {3}, be).relu().sum().at({0});
        data[i] -= 2.0 * eps;
        double fm = Tensor::from_data<double>(data, {3}, be).relu().sum().at({0});
        double num = (fp - fm) / (2.0 * eps);
        CHECK_NEAR(a.grad().at({i}), num, 1e-4);
    }
}

// ── graph properties ──────────────────────────────────────────────────────────

void test_math_ops_graph_properties() {
    std::cout << "[MathOps 27] graph: is_leaf false for op results with requires_grad input\n";
    Backend& be = cpu_backend();
    Tensor a = Tensor::from_data<double>({1.0, 2.0}, {2}, be, true);
    Tensor b = Tensor::from_data<double>({2.0, 3.0}, {2}, be, true);

    CHECK(!a.sub(b).is_leaf());
    CHECK(!a.div(b).is_leaf());
    CHECK(!a.neg().is_leaf());
    CHECK(!a.exp().is_leaf());
    CHECK(!a.log().is_leaf());
    // sqrt and relu need positive inputs to avoid nan
    Tensor pos = Tensor::from_data<double>({1.0, 4.0}, {2}, be, true);
    CHECK(!pos.sqrt().is_leaf());
    CHECK(!pos.relu().is_leaf());
}

void test_math_ops_no_grad_leaf() {
    std::cout << "[MathOps 28] graph: no grad graph built when no input requires_grad\n";
    Backend& be = cpu_backend();
    Tensor a = Tensor::from_data<double>({1.0, 2.0}, {2}, be);  // requires_grad=false
    Tensor b = Tensor::from_data<double>({3.0, 4.0}, {2}, be);

    Tensor r1 = a.sub(b);
    CHECK(r1.is_leaf());   // no requires_grad input → plain tensor, treated as leaf
    Tensor r2 = a.exp();
    CHECK(r2.is_leaf());
}

// ── composition ───────────────────────────────────────────────────────────────

void test_math_ops_composition_backward() {
    std::cout << "[MathOps 29] composition: log(exp(x)) backward = 1 everywhere\n";
    Backend& be = cpu_backend();
    // loss = sum(log(exp(a))), grad = 1 for all a (log and exp cancel)
    Tensor a = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be, true);
    Tensor loss = a.exp().log().sum();
    loss.backward();
    CHECK_NEAR(a.grad().at({0}), 1.0, 1e-6);
    CHECK_NEAR(a.grad().at({1}), 1.0, 1e-6);
    CHECK_NEAR(a.grad().at({2}), 1.0, 1e-6);
}

void test_math_ops_retain_graph() {
    std::cout << "[MathOps 30] retain_graph: two backward passes accumulate gradients\n";
    Backend& be = cpu_backend();
    Tensor a = Tensor::from_data<double>({2.0, 3.0}, {2}, be, true);
    Tensor loss = a.neg().sum();
    loss.backward(true);  // retain graph
    loss.backward(true);  // second pass — should accumulate
    // Each pass contributes -1 per element → total = -2
    CHECK_NEAR(a.grad().at({0}), -2.0, 1e-9);
    CHECK_NEAR(a.grad().at({1}), -2.0, 1e-9);
}

// ── memory ────────────────────────────────────────────────────────────────────

void test_math_ops_memory_clean() {
    std::cout << "[MathOps 31] memory: bytes_allocated == 0 after math ops and backward\n";
    {
        Backend& be = cpu_backend();
        Tensor a = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be, true);
        Tensor b = Tensor::from_data<double>({4.0, 5.0, 6.0}, {3}, be, true);
        a.neg().sum().backward();
        a.zero_grad();

        a.sub(b).sum().backward();
        a.zero_grad(); b.zero_grad();

        a.div(b).sum().backward();
        a.zero_grad(); b.zero_grad();

        a.exp().sum().backward();
        a.zero_grad();

        a.log().sum().backward();
        a.zero_grad();

        Tensor pos = Tensor::from_data<double>({1.0, 4.0, 9.0}, {3}, be, true);
        pos.sqrt().sum().backward();
        pos.zero_grad();

        pos.relu().sum().backward();
        pos.zero_grad();
    }
    CHECK(cpu_backend().memory_manager()->bytes_allocated() == 0);
}

// ── run_all ───────────────────────────────────────────────────────────────────

void run_math_ops_tests() {
    test_neg_forward();
    test_neg_backward();
    test_neg_no_grad_leaf();

    test_sub_forward();
    test_sub_backward();
    test_sub_broadcast_backward();

    test_div_forward();
    test_div_ieee754_edge_cases();
    test_div_backward();
    test_div_backward_numerical();

    test_exp_forward();
    test_exp_ieee754_edge_cases();
    test_exp_backward();
    test_exp_backward_numerical();

    test_log_forward();
    test_log_ieee754_edge_cases();
    test_log_backward();
    test_log_backward_numerical();

    test_sqrt_forward();
    test_sqrt_ieee754_edge_cases();
    test_sqrt_backward();
    test_sqrt_backward_zero_input();
    test_sqrt_backward_numerical();

    test_relu_forward();
    test_relu_backward_values();
    test_relu_backward_numerical();

    test_math_ops_graph_properties();
    test_math_ops_no_grad_leaf();
    test_math_ops_composition_backward();
    test_math_ops_retain_graph();
    test_math_ops_memory_clean();
}

} // namespace otter::test
