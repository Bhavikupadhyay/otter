#include "test_utils.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "otter/tensor.h"
#include "otter/backends/cpu.h"

// Tests for broadcast shape inference, stride computation, and BroadcastOp
// forward + backward wired through AddOperation.

namespace otter::test {

// ── Forward: broadcast add ────────────────────────────────────────────────────

void test_broadcast_add_row_to_matrix() {
    std::cout << "[Broadcast 1] add {3} + {2,3} → {2,3}: row broadcast\n";
    Backend& be = cpu_backend();
    // b = [10, 20, 30], a = [[1,2,3],[4,5,6]]
    // expected: [[11,22,33],[14,25,36]]
    Tensor a = Tensor::from_data<double>({1, 2, 3, 4, 5, 6}, {2, 3}, be);
    Tensor b = Tensor::from_data<double>({10, 20, 30},        {3},    be);
    Tensor c = a.add(b);
    CHECK(c.shape() == (std::vector<std::size_t>{2, 3}));
    CHECK_NEAR(c.at({0, 0}), 11.0, 1e-12);
    CHECK_NEAR(c.at({0, 1}), 22.0, 1e-12);
    CHECK_NEAR(c.at({0, 2}), 33.0, 1e-12);
    CHECK_NEAR(c.at({1, 0}), 14.0, 1e-12);
    CHECK_NEAR(c.at({1, 1}), 25.0, 1e-12);
    CHECK_NEAR(c.at({1, 2}), 36.0, 1e-12);
}

void test_broadcast_add_col_to_matrix() {
    std::cout << "[Broadcast 2] add {2,1} + {2,3} → {2,3}: column broadcast\n";
    Backend& be = cpu_backend();
    // a = [[1],[2]], b = [[10,20,30],[40,50,60]]
    // expected: [[11,21,31],[42,52,62]]
    Tensor a = Tensor::from_data<double>({1, 2},                {2, 1}, be);
    Tensor b = Tensor::from_data<double>({10, 20, 30, 40, 50, 60}, {2, 3}, be);
    Tensor c = a.add(b);
    CHECK(c.shape() == (std::vector<std::size_t>{2, 3}));
    CHECK_NEAR(c.at({0, 0}), 11.0, 1e-12);
    CHECK_NEAR(c.at({0, 1}), 21.0, 1e-12);
    CHECK_NEAR(c.at({0, 2}), 31.0, 1e-12);
    CHECK_NEAR(c.at({1, 0}), 42.0, 1e-12);
    CHECK_NEAR(c.at({1, 1}), 52.0, 1e-12);
    CHECK_NEAR(c.at({1, 2}), 62.0, 1e-12);
}

void test_broadcast_add_scalar_to_matrix() {
    std::cout << "[Broadcast 3] add {1} + {2,2} → {2,2}: scalar broadcast\n";
    Backend& be = cpu_backend();
    Tensor a = Tensor::from_data<double>({5},             {1},    be);
    Tensor b = Tensor::from_data<double>({1, 2, 3, 4},   {2, 2}, be);
    Tensor c = a.add(b);
    CHECK_NEAR(c.at({0, 0}), 6.0, 1e-12);
    CHECK_NEAR(c.at({0, 1}), 7.0, 1e-12);
    CHECK_NEAR(c.at({1, 0}), 8.0, 1e-12);
    CHECK_NEAR(c.at({1, 1}), 9.0, 1e-12);
}

void test_broadcast_add_incompatible_throws() {
    std::cout << "[Broadcast 4] add {2,3} + {2,4} — incompatible shapes throw\n";
    Backend& be = cpu_backend();
    Tensor a = Tensor::zeros({2, 3}, be);
    Tensor b = Tensor::zeros({2, 4}, be);
    bool threw = false;
    try {
        Tensor c = a.add(b);
        (void)c;
    } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

// ── Backward: gradient reduces to original shape ──────────────────────────────

void test_broadcast_backward_row_to_matrix() {
    std::cout << "[Broadcast 5] backward {3}+{2,3}: row grad reduces over batch dim\n";
    Backend& be = cpu_backend();
    // a (requires_grad) = [1,2,3], b = [[4,5,6],[7,8,9]]
    // loss = sum(a + b) = sum([[5,7,9],[8,10,12]]) = 51
    // d loss/d a_j = sum over rows = 2 (appears in 2 rows), so grad_a = [2,2,2]
    Tensor a = Tensor::from_data<double>({1, 2, 3},          {3},    be, /*requires_grad=*/true);
    Tensor b = Tensor::from_data<double>({4, 5, 6, 7, 8, 9}, {2, 3}, be);
    a.add(b).sum().backward();
    CHECK_NEAR(a.grad().at({0}), 2.0, 1e-12);  // row 0 + row 1
    CHECK_NEAR(a.grad().at({1}), 2.0, 1e-12);
    CHECK_NEAR(a.grad().at({2}), 2.0, 1e-12);
}

void test_broadcast_backward_col_to_matrix() {
    std::cout << "[Broadcast 6] backward {2,1}+{2,3}: col grad reduces over cols\n";
    Backend& be = cpu_backend();
    // a (requires_grad) = [[1],[2]], b = [[10,20,30],[40,50,60]]
    // d loss/d a_i0 = 3 (appears in 3 columns), so grad_a = [[3],[3]]
    Tensor a = Tensor::from_data<double>({1, 2},                {2, 1}, be, /*requires_grad=*/true);
    Tensor b = Tensor::from_data<double>({10, 20, 30, 40, 50, 60}, {2, 3}, be);
    a.add(b).sum().backward();
    CHECK_NEAR(a.grad().at({0, 0}), 3.0, 1e-12);  // sums over 3 cols
    CHECK_NEAR(a.grad().at({1, 0}), 3.0, 1e-12);
}

void test_broadcast_backward_both_inputs_need_reduction() {
    std::cout << "[Broadcast 7] backward {3}+{2,3}: both a and b require_grad\n";
    Backend& be = cpu_backend();
    Tensor a = Tensor::from_data<double>({1, 2, 3},          {3},    be, /*requires_grad=*/true);
    Tensor b = Tensor::from_data<double>({4, 5, 6, 7, 8, 9}, {2, 3}, be, /*requires_grad=*/true);
    a.add(b).sum().backward();
    // grad_a: each element appears in 2 rows → grad = [2,2,2]
    CHECK_NEAR(a.grad().at({0}), 2.0, 1e-12);
    CHECK_NEAR(a.grad().at({1}), 2.0, 1e-12);
    CHECK_NEAR(a.grad().at({2}), 2.0, 1e-12);
    // grad_b: shape {2,3}, each element appears once → grad = all 1s
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            CHECK_NEAR(b.grad().at({r, c}), 1.0, 1e-12);
}

// ── Run all ───────────────────────────────────────────────────────────────────

void run_broadcast_tests() {
    test_broadcast_add_row_to_matrix();
    test_broadcast_add_col_to_matrix();
    test_broadcast_add_scalar_to_matrix();
    test_broadcast_add_incompatible_throws();
    test_broadcast_backward_row_to_matrix();
    test_broadcast_backward_col_to_matrix();
    test_broadcast_backward_both_inputs_need_reduction();
}

} // namespace otter::test
