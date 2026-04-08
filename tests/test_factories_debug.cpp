#include "test_utils.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "otter/tensor.h"
#include "otter/debug.h"
#include "otter/backends/cpu.h"

// Tests for new factory methods (ones, full, zeros_like, ones_like),
// to_vector<T>(), print(), and debug utilities (has_nan, has_inf, max_abs_diff,
// shape_str, dtype_str).

namespace otter::test {

// ── ones ──────────────────────────────────────────────────────────────────────

void test_ones_values() {
    std::cout << "[FD 1] ones: all elements == 1.0\n";
    Backend& be = cpu_backend();
    Tensor t = Tensor::ones({2, 3}, be);
    CHECK(t.shape() == (std::vector<std::size_t>{2, 3}));
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            CHECK_NEAR(t.at({i, j}), 1.0, 1e-12);
}

void test_ones_requires_grad() {
    std::cout << "[FD 2] ones: requires_grad=true propagates\n";
    Backend& be = cpu_backend();
    Tensor t = Tensor::ones({3}, be, DType::Float64, true);
    CHECK(t.requires_grad());
    CHECK(t.is_leaf());
}

// ── full ──────────────────────────────────────────────────────────────────────

void test_full_values() {
    std::cout << "[FD 3] full: all elements == given value\n";
    Backend& be = cpu_backend();
    Tensor t = Tensor::full({2, 2}, 7.5, be);
    CHECK(t.shape() == (std::vector<std::size_t>{2, 2}));
    CHECK_NEAR(t.at({0, 0}), 7.5, 1e-12);
    CHECK_NEAR(t.at({0, 1}), 7.5, 1e-12);
    CHECK_NEAR(t.at({1, 0}), 7.5, 1e-12);
    CHECK_NEAR(t.at({1, 1}), 7.5, 1e-12);
}

void test_full_negative_value() {
    std::cout << "[FD 4] full: negative fill value\n";
    Backend& be = cpu_backend();
    Tensor t = Tensor::full({4}, -3.0, be);
    for (std::size_t i = 0; i < 4; ++i)
        CHECK_NEAR(t.at({i}), -3.0, 1e-12);
}

// ── zeros_like / ones_like ────────────────────────────────────────────────────

void test_zeros_like_shape_dtype_backend() {
    std::cout << "[FD 5] zeros_like: matches shape, dtype, backend of source tensor\n";
    Backend& be = cpu_backend();
    Tensor src = Tensor::from_data<double>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3}, be);
    Tensor z   = Tensor::zeros_like(src);
    CHECK(z.shape() == src.shape());
    CHECK(z.dtype() == src.dtype());
    CHECK(!z.requires_grad());
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            CHECK_NEAR(z.at({i, j}), 0.0, 1e-12);
}

void test_zeros_like_independent_buffer() {
    std::cout << "[FD 6] zeros_like: result is independent (separate buffer)\n";
    Backend& be = cpu_backend();
    Tensor src = Tensor::ones({3}, be);
    Tensor z   = Tensor::zeros_like(src);
    // src still has ones, z has zeros — independent allocations
    CHECK_NEAR(src.at({0}), 1.0, 1e-12);
    CHECK_NEAR(z.at({0}),   0.0, 1e-12);
}

void test_ones_like_values() {
    std::cout << "[FD 7] ones_like: all elements == 1.0, same shape as source\n";
    Backend& be = cpu_backend();
    Tensor src = Tensor::zeros({3, 2}, be);
    Tensor o   = Tensor::ones_like(src);
    CHECK(o.shape() == src.shape());
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            CHECK_NEAR(o.at({i, j}), 1.0, 1e-12);
}

void test_ones_like_requires_grad() {
    std::cout << "[FD 8] ones_like: requires_grad=true propagates\n";
    Backend& be = cpu_backend();
    Tensor src = Tensor::zeros({4}, be);
    Tensor o   = Tensor::ones_like(src, true);
    CHECK(o.requires_grad());
}

// ── to_vector ─────────────────────────────────────────────────────────────────

void test_to_vector_contiguous() {
    std::cout << "[FD 9] to_vector: contiguous tensor matches input data\n";
    Backend& be = cpu_backend();
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    Tensor t = Tensor::from_data<double>(data, {2, 3}, be);
    auto v   = t.to_vector<double>();
    CHECK(v.size() == data.size());
    for (std::size_t i = 0; i < data.size(); ++i)
        CHECK_NEAR(v[i], data[i], 1e-12);
}

void test_to_vector_after_transpose() {
    std::cout << "[FD 10] to_vector: non-contiguous (transposed) tensor reads correctly\n";
    Backend& be = cpu_backend();
    // [[1,2,3],[4,5,6]] transposed → [[1,4],[2,5],[3,6]]
    Tensor t  = Tensor::from_data<double>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3}, be);
    Tensor tr = t.transpose(0, 1);  // shape {3, 2}
    auto v    = tr.to_vector<double>();
    CHECK(v.size() == 6);
    // Row-major of transposed: [1,4, 2,5, 3,6]
    CHECK_NEAR(v[0], 1.0, 1e-12);
    CHECK_NEAR(v[1], 4.0, 1e-12);
    CHECK_NEAR(v[2], 2.0, 1e-12);
    CHECK_NEAR(v[3], 5.0, 1e-12);
    CHECK_NEAR(v[4], 3.0, 1e-12);
    CHECK_NEAR(v[5], 6.0, 1e-12);
}

void test_to_vector_scalar() {
    std::cout << "[FD 11] to_vector: scalar tensor {1} returns single-element vector\n";
    Backend& be = cpu_backend();
    Tensor t = Tensor::from_data<double>({42.0}, {1}, be);
    auto v   = t.to_vector<double>();
    CHECK(v.size() == 1);
    CHECK_NEAR(v[0], 42.0, 1e-12);
}

// ── has_nan ───────────────────────────────────────────────────────────────────

void test_has_nan_false() {
    std::cout << "[FD 12] has_nan: returns false for clean tensor\n";
    Backend& be = cpu_backend();
    Tensor t = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be);
    CHECK(!has_nan(t));
}

void test_has_nan_true() {
    std::cout << "[FD 13] has_nan: returns true when any element is nan\n";
    Backend& be = cpu_backend();
    // log(-1) produces nan
    Tensor t = Tensor::from_data<double>({1.0, -1.0, 3.0}, {3}, be);
    Tensor r = t.log();
    CHECK(has_nan(r));  // log(-1) = nan
}

// ── has_inf ───────────────────────────────────────────────────────────────────

void test_has_inf_false() {
    std::cout << "[FD 14] has_inf: returns false for clean tensor\n";
    Backend& be = cpu_backend();
    Tensor t = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be);
    CHECK(!has_inf(t));
}

void test_has_inf_true() {
    std::cout << "[FD 15] has_inf: returns true when any element is ±inf\n";
    Backend& be = cpu_backend();
    // 1 / 0 produces +inf
    Tensor a = Tensor::from_data<double>({1.0, 2.0}, {2}, be);
    Tensor b = Tensor::from_data<double>({0.0, 1.0}, {2}, be);
    Tensor r = a.div(b);
    CHECK(has_inf(r));  // r[0] = +inf
}

// ── max_abs_diff ──────────────────────────────────────────────────────────────

void test_max_abs_diff_zero() {
    std::cout << "[FD 16] max_abs_diff: identical tensors → 0.0\n";
    Backend& be = cpu_backend();
    Tensor a = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be);
    Tensor b = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be);
    CHECK_NEAR(max_abs_diff(a, b), 0.0, 1e-15);
}

void test_max_abs_diff_value() {
    std::cout << "[FD 17] max_abs_diff: returns correct maximum absolute difference\n";
    Backend& be = cpu_backend();
    // differences: |1-1|=0, |4-2|=2, |3-6|=3 → max = 3
    Tensor a = Tensor::from_data<double>({1.0, 4.0, 3.0}, {3}, be);
    Tensor b = Tensor::from_data<double>({1.0, 2.0, 6.0}, {3}, be);
    CHECK_NEAR(max_abs_diff(a, b), 3.0, 1e-12);
}

void test_max_abs_diff_shape_mismatch_throws() {
    std::cout << "[FD 18] max_abs_diff: shape mismatch throws\n";
    Backend& be = cpu_backend();
    Tensor a = Tensor::from_data<double>({1.0, 2.0}, {2}, be);
    Tensor b = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be);
    bool threw = false;
    try { max_abs_diff(a, b); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

// ── shape_str / dtype_str ─────────────────────────────────────────────────────

void test_shape_str() {
    std::cout << "[FD 19] shape_str: correct string for {2,3} tensor\n";
    Backend& be = cpu_backend();
    Tensor t = Tensor::zeros({2, 3}, be);
    CHECK(shape_str(t) == "[2, 3]");
}

void test_dtype_str() {
    std::cout << "[FD 20] dtype_str: Float64 tensor returns \"Float64\"\n";
    Backend& be = cpu_backend();
    Tensor t = Tensor::zeros({2}, be);
    CHECK(dtype_str(t) == "Float64");
}

// ── memory ────────────────────────────────────────────────────────────────────

void test_factories_memory_clean() {
    std::cout << "[FD 21] memory: bytes_allocated == 0 after all factory tensors destruct\n";
    {
        Backend& be = cpu_backend();
        Tensor a = Tensor::ones({3, 4}, be);
        Tensor b = Tensor::full({2, 2}, 5.0, be);
        Tensor c = Tensor::zeros_like(a);
        Tensor d = Tensor::ones_like(b);
        auto   v = a.to_vector<double>();
        (void)v;
    }
    CHECK(cpu_backend().memory_manager()->bytes_allocated() == 0);
}

// ── run_all ───────────────────────────────────────────────────────────────────

void run_factories_debug_tests() {
    test_ones_values();
    test_ones_requires_grad();

    test_full_values();
    test_full_negative_value();

    test_zeros_like_shape_dtype_backend();
    test_zeros_like_independent_buffer();
    test_ones_like_values();
    test_ones_like_requires_grad();

    test_to_vector_contiguous();
    test_to_vector_after_transpose();
    test_to_vector_scalar();

    test_has_nan_false();
    test_has_nan_true();
    test_has_inf_false();
    test_has_inf_true();

    test_max_abs_diff_zero();
    test_max_abs_diff_value();
    test_max_abs_diff_shape_mismatch_throws();

    test_shape_str();
    test_dtype_str();

    test_factories_memory_clean();
}

} // namespace otter::test
