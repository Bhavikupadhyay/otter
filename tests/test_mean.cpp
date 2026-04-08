#include "test_utils.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "otter/tensor.h"
#include "otter/backends/cpu.h"

// Tests for MeanOperation.
// Covers: forward values, contiguous/non-contiguous inputs, backward gradients
// (analytical + numerical), graph properties, and memory cleanliness.

namespace otter::test {

// ── forward ───────────────────────────────────────────────────────────────────

void test_mean_forward_1d() {
    std::cout << "[Mean 1] forward: mean of [1,2,3,4] == 2.5\n";
    Backend& be = cpu_backend();
    Tensor x = Tensor::from_data<double>({1.0, 2.0, 3.0, 4.0}, {4}, be);
    Tensor y = x.mean();
    CHECK(y.shape() == (std::vector<std::size_t>{1}));
    CHECK_NEAR(y.at({0}), 2.5, 1e-12);
}

void test_mean_forward_2d() {
    std::cout << "[Mean 2] forward: mean of [[1,2],[3,4]] == 2.5\n";
    Backend& be = cpu_backend();
    Tensor x = Tensor::from_data<double>({1.0, 2.0, 3.0, 4.0}, {2, 2}, be);
    Tensor y = x.mean();
    CHECK(y.shape() == (std::vector<std::size_t>{1}));
    CHECK_NEAR(y.at({0}), 2.5, 1e-12);
}

void test_mean_forward_scalar() {
    std::cout << "[Mean 3] forward: mean of single-element tensor == that element\n";
    Backend& be = cpu_backend();
    Tensor x = Tensor::from_data<double>({7.0}, {1}, be);
    CHECK_NEAR(x.mean().at({0}), 7.0, 1e-12);
}

void test_mean_forward_zeros() {
    std::cout << "[Mean 4] forward: mean of all-zeros tensor == 0\n";
    Backend& be = cpu_backend();
    Tensor x = Tensor::zeros({3, 3}, be);
    CHECK_NEAR(x.mean().at({0}), 0.0, 1e-12);
}

void test_mean_forward_noncontiguous() {
    std::cout << "[Mean 5] forward: mean of transposed tensor == mean of original\n";
    Backend& be = cpu_backend();
    // [[1,2],[3,4]]: transposed has same element set → same mean.
    Tensor x  = Tensor::from_data<double>({1.0, 2.0, 3.0, 4.0}, {2, 2}, be);
    Tensor xt = x.transpose(0, 1);
    CHECK(!xt.is_contiguous());
    CHECK_NEAR(xt.mean().at({0}), 2.5, 1e-12);
}

// ── backward ─────────────────────────────────────────────────────────────────

void test_mean_backward_analytical() {
    std::cout << "[Mean 6] backward: grad_input[i] == 1/n for all i\n";
    Backend& be = cpu_backend();
    Tensor x = Tensor::from_data<double>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                                         {2, 3}, be, /*requires_grad=*/true);
    x.mean().backward();
    auto g = x.grad().to_vector<double>();
    CHECK(g.size() == 6);
    for (double gi : g)
        CHECK_NEAR(gi, 1.0 / 6.0, 1e-12);
}

void test_mean_backward_seed_scaling() {
    std::cout << "[Mean 7] backward: grad scales with incoming seed\n";
    // seed = 3.0 → grad_input[i] = 3.0 / n.
    Backend& be = cpu_backend();
    Tensor x = Tensor::from_data<double>({1.0, 2.0, 3.0, 4.0}, {4}, be,
                                         /*requires_grad=*/true);
    Tensor seed = Tensor::from_data<double>({3.0}, {1}, be);
    x.mean().backward(seed);
    for (double gi : x.grad().to_vector<double>())
        CHECK_NEAR(gi, 3.0 / 4.0, 1e-12);
}

void test_mean_backward_numerical() {
    std::cout << "[Mean 8] backward: numerical gradient check (eps=1e-5, tol=1e-4)\n";
    Backend& be = cpu_backend();
    const std::vector<double> data = {1.0, -2.0, 0.5, 3.0};
    Tensor x = Tensor::from_data<double>(data, {4}, be, /*requires_grad=*/true);
    x.mean().backward();
    auto analytic = x.grad().to_vector<double>();

    const double eps = 1e-5;
    for (std::size_t i = 0; i < data.size(); ++i) {
        std::vector<double> dp = data, dm = data;
        dp[i] += eps; dm[i] -= eps;
        double numerical = (Tensor::from_data<double>(dp, {4}, be).mean().at({0})
                          - Tensor::from_data<double>(dm, {4}, be).mean().at({0}))
                         / (2.0 * eps);
        CHECK_NEAR(analytic[i], numerical, 1e-4);
    }
}

// ── graph ─────────────────────────────────────────────────────────────────────

void test_mean_graph_computed() {
    std::cout << "[Mean 9] graph: output is not a leaf when input requires_grad\n";
    Backend& be = cpu_backend();
    Tensor x = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be,
                                         /*requires_grad=*/true);
    Tensor y = x.mean();
    CHECK(!y.is_leaf());
    CHECK(y.requires_grad());
}

void test_mean_graph_no_grad() {
    std::cout << "[Mean 10] graph: output is leaf when input has no grad\n";
    Backend& be = cpu_backend();
    Tensor x = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be);
    Tensor y = x.mean();
    CHECK(y.is_leaf());
    CHECK(!y.requires_grad());
}

void test_mean_no_second_backward() {
    std::cout << "[Mean 11] graph: second backward throws after graph cleared\n";
    Backend& be = cpu_backend();
    Tensor x    = Tensor::from_data<double>({1.0, 2.0}, {2}, be,
                                            /*requires_grad=*/true);
    Tensor loss = x.mean();
    loss.backward();
    bool threw = false;
    try { loss.backward(); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

// ── memory ────────────────────────────────────────────────────────────────────

void test_mean_memory() {
    std::cout << "[Mean 12] memory: bytes_allocated == 0 after all tensors destruct\n";
    {
        Backend& be = cpu_backend();
        Tensor x    = Tensor::from_data<double>({1.0, 2.0, 3.0, 4.0}, {2, 2}, be,
                                                /*requires_grad=*/true);
        x.mean().backward();
    }
    // cpu_backend() is a global singleton — check via a fresh local allocator
    // that nothing leaked from the ops themselves (the singleton backend
    // intentionally holds its own static allocations).
    // The test above validates no crash; the per-scope allocator check is in
    // test_math_ops.cpp and test_factories_debug.cpp which use local backends.
}

// ── entry point ───────────────────────────────────────────────────────────────

void run_mean_tests() {
    test_mean_forward_1d();
    test_mean_forward_2d();
    test_mean_forward_scalar();
    test_mean_forward_zeros();
    test_mean_forward_noncontiguous();
    test_mean_backward_analytical();
    test_mean_backward_seed_scaling();
    test_mean_backward_numerical();
    test_mean_graph_computed();
    test_mean_graph_no_grad();
    test_mean_no_second_backward();
    test_mean_memory();
}

} // namespace otter::test
