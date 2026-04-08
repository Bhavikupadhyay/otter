#include "test_utils.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "otter/tensor.h"
#include "otter/autograd/no_grad_guard.h"
#include "otter/backends/cpu.h"

// Tests for AddOperation, SumOperation, and Tensor::backward().
// Every expected gradient value is derived from first principles and documented.

namespace otter::test {

// ── AddOperation forward ──────────────────────────────────────────────────────

void test_add_forward_values() {
    std::cout << "[Autograd 1] add: {2,2} + {2,2} — correct values\n";
    Backend& be = cpu_backend();
    // a = [[1,2],[3,4]], b = [[10,20],[30,40]]
    Tensor a = Tensor::from_data<double>({1, 2, 3, 4},       {2, 2}, be);
    Tensor b = Tensor::from_data<double>({10, 20, 30, 40},   {2, 2}, be);
    Tensor c = a.add(b);
    CHECK_NEAR(c.at({0, 0}), 11.0, 1e-12);
    CHECK_NEAR(c.at({0, 1}), 22.0, 1e-12);
    CHECK_NEAR(c.at({1, 0}), 33.0, 1e-12);
    CHECK_NEAR(c.at({1, 1}), 44.0, 1e-12);
}

void test_add_output_not_leaf() {
    std::cout << "[Autograd 2] add: output is non-leaf when any input requires_grad\n";
    Backend& be = cpu_backend();
    Tensor a = Tensor::from_data<double>({1, 2}, {2}, be, /*requires_grad=*/true);
    Tensor b = Tensor::from_data<double>({3, 4}, {2}, be, /*requires_grad=*/false);
    Tensor c = a.add(b);
    CHECK(c.requires_grad());
    CHECK(!c.is_leaf());
}

void test_add_no_grad_when_inputs_detached() {
    std::cout << "[Autograd 3] add: no grad graph when both inputs have requires_grad=false\n";
    Backend& be = cpu_backend();
    Tensor a = Tensor::from_data<double>({1, 2}, {2}, be);
    Tensor b = Tensor::from_data<double>({3, 4}, {2}, be);
    Tensor c = a.add(b);
    CHECK(!c.requires_grad());
    CHECK(c.is_leaf());
}

// ── SumOperation forward ─────────────────────────────────────────────────────

void test_sum_forward_value() {
    std::cout << "[Autograd 4] sum: {2,3} → scalar == 21\n";
    Backend& be = cpu_backend();
    // [[1,2,3],[4,5,6]] → sum = 21
    Tensor a = Tensor::from_data<double>({1, 2, 3, 4, 5, 6}, {2, 3}, be);
    Tensor s = a.sum();
    CHECK(s.shape() == std::vector<std::size_t>{1});
    CHECK_NEAR(s.at({0}), 21.0, 1e-12);
}

// ── Backward: add + sum ───────────────────────────────────────────────────────

void test_add_backward_gradients() {
    std::cout << "[Autograd 5] add backward: grad_a == grad_b == ones (same shape)\n";
    Backend& be = cpu_backend();
    Tensor a = Tensor::from_data<double>({1, 2, 3}, {3}, be, /*requires_grad=*/true);
    Tensor b = Tensor::from_data<double>({4, 5, 6}, {3}, be, /*requires_grad=*/true);
    Tensor loss = a.add(b).sum();
    loss.backward();
    // d(sum(a+b))/da_i = 1, d(sum(a+b))/db_i = 1 for all i
    for (std::size_t i = 0; i < 3; ++i) {
        CHECK_NEAR(a.grad().at({i}), 1.0, 1e-12);
        CHECK_NEAR(b.grad().at({i}), 1.0, 1e-12);
    }
}

void test_sum_backward_fans_scalar() {
    std::cout << "[Autograd 6] sum backward: scalar grad fans out to all input elements\n";
    Backend& be = cpu_backend();
    Tensor a = Tensor::from_data<double>({2, 4, 6, 8}, {2, 2}, be, /*requires_grad=*/true);
    Tensor loss = a.sum();  // grad at loss = 1.0 (seeded by backward())
    loss.backward();
    // d(sum(a))/da_ij = 1 for all i,j
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 2; ++c)
            CHECK_NEAR(a.grad().at({r, c}), 1.0, 1e-12);
}

void test_add_chain_backward() {
    std::cout << "[Autograd 7] chain: (a + b + c).sum() — all three grads == 1\n";
    Backend& be = cpu_backend();
    Tensor a = Tensor::from_data<double>({1, 2}, {2}, be, /*requires_grad=*/true);
    Tensor b = Tensor::from_data<double>({3, 4}, {2}, be, /*requires_grad=*/true);
    Tensor c = Tensor::from_data<double>({5, 6}, {2}, be, /*requires_grad=*/true);
    // loss = sum(a + b + c) = sum((a+b)+c)
    Tensor loss = a.add(b).add(c).sum();
    loss.backward();
    for (std::size_t i = 0; i < 2; ++i) {
        CHECK_NEAR(a.grad().at({i}), 1.0, 1e-12);
        CHECK_NEAR(b.grad().at({i}), 1.0, 1e-12);
        CHECK_NEAR(c.grad().at({i}), 1.0, 1e-12);
    }
}

// ── zero_grad ─────────────────────────────────────────────────────────────────

void test_zero_grad_clears_after_backward() {
    std::cout << "[Autograd 8] zero_grad: clears accumulated gradient\n";
    Backend& be = cpu_backend();
    Tensor a = Tensor::from_data<double>({1, 2, 3}, {3}, be, /*requires_grad=*/true);
    a.add(Tensor::from_data<double>({4, 5, 6}, {3}, be)).sum().backward();
    CHECK(a.grad().defined());
    a.zero_grad();
    CHECK(!a.grad().defined());
}

// ── retain_graph ──────────────────────────────────────────────────────────────

void test_retain_graph_accumulates() {
    std::cout << "[Autograd 9] retain_graph: two backward passes accumulate gradients\n";
    Backend& be = cpu_backend();
    Tensor a = Tensor::from_data<double>({1, 2}, {2}, be, /*requires_grad=*/true);
    Tensor b = Tensor::from_data<double>({3, 4}, {2}, be);
    Tensor loss = a.add(b).sum();

    loss.backward(/*retain_graph=*/true);
    // After 1st backward: grad_a = [1, 1]
    CHECK_NEAR(a.grad().at({0}), 1.0, 1e-12);
    CHECK_NEAR(a.grad().at({1}), 1.0, 1e-12);

    loss.backward(/*retain_graph=*/false);
    // After 2nd backward: grad_a = [2, 2] (accumulated, not reset)
    CHECK_NEAR(a.grad().at({0}), 2.0, 1e-12);
    CHECK_NEAR(a.grad().at({1}), 2.0, 1e-12);
}

void test_second_backward_after_cleanup_throws() {
    std::cout << "[Autograd 10] backward twice without retain_graph throws\n";
    Backend& be = cpu_backend();
    Tensor a = Tensor::from_data<double>({1, 2}, {2}, be, /*requires_grad=*/true);
    Tensor loss = a.add(Tensor::from_data<double>({3, 4}, {2}, be)).sum();
    loss.backward();
    bool threw = false;
    try { loss.backward(); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

// ── NoGradGuard skips graph wiring ────────────────────────────────────────────

void test_no_grad_guard_skips_graph() {
    std::cout << "[Autograd 11] NoGradGuard: output has no grad graph\n";
    Backend& be = cpu_backend();
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

// ── detach severs gradient flow ───────────────────────────────────────────────

void test_detach_severs_grad_flow() {
    std::cout << "[Autograd 12] detach: gradient does not flow through detached tensor\n";
    Backend& be = cpu_backend();
    Tensor a = Tensor::from_data<double>({1, 2}, {2}, be, /*requires_grad=*/true);
    Tensor b = Tensor::from_data<double>({3, 4}, {2}, be, /*requires_grad=*/true);
    // Detach b before the add — grad should flow to a but not b
    Tensor loss = a.add(b.detach()).sum();
    loss.backward();
    // a receives gradient (still in graph)
    CHECK(a.grad().defined());
    CHECK_NEAR(a.grad().at({0}), 1.0, 1e-12);
    CHECK_NEAR(a.grad().at({1}), 1.0, 1e-12);
    // b does not (detached from graph)
    CHECK(!b.grad().defined());
}

// ── bytes_allocated == 0 after scope ─────────────────────────────────────────

void test_autograd_memory_clean() {
    std::cout << "[Autograd 13] memory: bytes_allocated == 0 after all tensors destruct\n";
    Backend& be = cpu_backend();
    {
        Tensor a = Tensor::from_data<double>({1, 2, 3}, {3}, be, /*requires_grad=*/true);
        Tensor b = Tensor::from_data<double>({4, 5, 6}, {3}, be, /*requires_grad=*/true);
        a.add(b).sum().backward();
    }
    CHECK(be.memory_manager()->bytes_allocated() == 0);
}

// ── MulOperation forward ─────────────────────────────────────────────────────

void test_mul_forward_values() {
    std::cout << "[Autograd 14] mul: {2,2} * {2,2} — correct values\n";
    Backend& be = cpu_backend();
    // a = [[1,2],[3,4]], b = [[10,20],[30,40]]
    Tensor a = Tensor::from_data<double>({1, 2, 3, 4},     {2, 2}, be);
    Tensor b = Tensor::from_data<double>({10, 20, 30, 40}, {2, 2}, be);
    Tensor c = a.mul(b);
    CHECK_NEAR(c.at({0, 0}),  10.0, 1e-12);
    CHECK_NEAR(c.at({0, 1}),  40.0, 1e-12);
    CHECK_NEAR(c.at({1, 0}),  90.0, 1e-12);
    CHECK_NEAR(c.at({1, 1}), 160.0, 1e-12);
}

void test_mul_backward_gradients() {
    std::cout << "[Autograd 15] mul backward: grad_a = b, grad_b = a (product rule)\n";
    Backend& be = cpu_backend();
    // loss = sum(a * b); d loss/d a_i = b_i, d loss/d b_i = a_i
    Tensor a = Tensor::from_data<double>({2, 3, 4}, {3}, be, /*requires_grad=*/true);
    Tensor b = Tensor::from_data<double>({5, 6, 7}, {3}, be, /*requires_grad=*/true);
    a.mul(b).sum().backward();
    // grad_a = b = [5, 6, 7]
    CHECK_NEAR(a.grad().at({0}), 5.0, 1e-12);
    CHECK_NEAR(a.grad().at({1}), 6.0, 1e-12);
    CHECK_NEAR(a.grad().at({2}), 7.0, 1e-12);
    // grad_b = a = [2, 3, 4]
    CHECK_NEAR(b.grad().at({0}), 2.0, 1e-12);
    CHECK_NEAR(b.grad().at({1}), 3.0, 1e-12);
    CHECK_NEAR(b.grad().at({2}), 4.0, 1e-12);
}

void test_add_mul_chain_backward() {
    std::cout << "[Autograd 16] chain: ((a + b) * c).sum() — product + add rule\n";
    Backend& be = cpu_backend();
    // loss = sum((a+b)*c)
    // d loss/d a_i = c_i
    // d loss/d b_i = c_i
    // d loss/d c_i = (a_i + b_i)
    Tensor a = Tensor::from_data<double>({1, 2}, {2}, be, /*requires_grad=*/true);
    Tensor b = Tensor::from_data<double>({3, 4}, {2}, be, /*requires_grad=*/true);
    Tensor c = Tensor::from_data<double>({5, 6}, {2}, be, /*requires_grad=*/true);
    a.add(b).mul(c).sum().backward();
    // grad_a = c = [5, 6]
    CHECK_NEAR(a.grad().at({0}), 5.0, 1e-12);
    CHECK_NEAR(a.grad().at({1}), 6.0, 1e-12);
    // grad_b = c = [5, 6]
    CHECK_NEAR(b.grad().at({0}), 5.0, 1e-12);
    CHECK_NEAR(b.grad().at({1}), 6.0, 1e-12);
    // grad_c = (a + b) = [4, 6]
    CHECK_NEAR(c.grad().at({0}), 4.0, 1e-12);
    CHECK_NEAR(c.grad().at({1}), 6.0, 1e-12);
}

// ── Run all ───────────────────────────────────────────────────────────────────

void run_autograd_tests() {
    test_add_forward_values();
    test_add_output_not_leaf();
    test_add_no_grad_when_inputs_detached();
    test_sum_forward_value();
    test_add_backward_gradients();
    test_sum_backward_fans_scalar();
    test_add_chain_backward();
    test_zero_grad_clears_after_backward();
    test_retain_graph_accumulates();
    test_second_backward_after_cleanup_throws();
    test_no_grad_guard_skips_graph();
    test_detach_severs_grad_flow();
    test_autograd_memory_clean();
    test_mul_forward_values();
    test_mul_backward_gradients();
    test_add_mul_chain_backward();
}

} // namespace otter::test
