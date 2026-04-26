#pragma once

#include <cstddef>
#include <vector>

#include "../utils/test_utils.h"
#include "otter/tensor.h"
#include "otter/debug.h"
#include "otter/kernel/backend.h"

namespace otter::test::shared {

// ── ones ─────────────────────────────────────────────────────────────────────

inline void test_ones_values(Backend& be) {
    std::cout << "[Factories 1] ones: all elements == 1.0\n";
    Tensor t = Tensor::ones({2, 3}, be);
    CHECK(t.shape() == (std::vector<std::size_t>{2, 3}));
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            CHECK_NEAR(t.at({i, j}), 1.0, 1e-12);
}

inline void test_ones_requires_grad(Backend& be) {
    std::cout << "[Factories 2] ones: requires_grad=true propagates\n";
    Tensor t = Tensor::ones({3}, be, DType::Float64, true);
    CHECK(t.requires_grad());
    CHECK(t.is_leaf());
}

// ── full ─────────────────────────────────────────────────────────────────────

inline void test_full_values(Backend& be) {
    std::cout << "[Factories 3] full: all elements == given value\n";
    Tensor t = Tensor::full({2, 2}, 7.5, be);
    CHECK(t.shape() == (std::vector<std::size_t>{2, 2}));
    CHECK_NEAR(t.at({0, 0}), 7.5, 1e-12);
    CHECK_NEAR(t.at({0, 1}), 7.5, 1e-12);
    CHECK_NEAR(t.at({1, 0}), 7.5, 1e-12);
    CHECK_NEAR(t.at({1, 1}), 7.5, 1e-12);
}

inline void test_full_negative_value(Backend& be) {
    std::cout << "[Factories 4] full: negative fill value\n";
    Tensor t = Tensor::full({4}, -3.0, be);
    for (std::size_t i = 0; i < 4; ++i)
        CHECK_NEAR(t.at({i}), -3.0, 1e-12);
}

// ── zeros_like / ones_like ───────────────────────────────────────────────────

inline void test_zeros_like_shape_dtype_backend(Backend& be) {
    std::cout << "[Factories 5] zeros_like: matches shape, dtype, backend of source\n";
    Tensor src = Tensor::from_data<double>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3}, be);
    Tensor z   = Tensor::zeros_like(src);
    CHECK(z.shape() == src.shape());
    CHECK(z.dtype() == src.dtype());
    CHECK(!z.requires_grad());
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            CHECK_NEAR(z.at({i, j}), 0.0, 1e-12);
}

inline void test_zeros_like_independent_buffer(Backend& be) {
    std::cout << "[Factories 6] zeros_like: result is independent (separate buffer)\n";
    Tensor src = Tensor::ones({3}, be);
    Tensor z   = Tensor::zeros_like(src);
    CHECK_NEAR(src.at({0}), 1.0, 1e-12);
    CHECK_NEAR(z.at({0}),   0.0, 1e-12);
}

inline void test_ones_like_values(Backend& be) {
    std::cout << "[Factories 7] ones_like: all elements == 1.0, same shape as source\n";
    Tensor src = Tensor::zeros({3, 2}, be);
    Tensor o   = Tensor::ones_like(src);
    CHECK(o.shape() == src.shape());
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            CHECK_NEAR(o.at({i, j}), 1.0, 1e-12);
}

inline void test_ones_like_requires_grad(Backend& be) {
    std::cout << "[Factories 8] ones_like: requires_grad=true propagates\n";
    Tensor src = Tensor::zeros({4}, be);
    Tensor o   = Tensor::ones_like(src, true);
    CHECK(o.requires_grad());
}

// ── to_vector ────────────────────────────────────────────────────────────────

inline void test_to_vector_contiguous(Backend& be) {
    std::cout << "[Factories 9] to_vector: contiguous tensor matches input data\n";
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    Tensor t = Tensor::from_data<double>(data, {2, 3}, be);
    auto v   = t.to_vector<double>();
    CHECK(v.size() == data.size());
    for (std::size_t i = 0; i < data.size(); ++i)
        CHECK_NEAR(v[i], data[i], 1e-12);
}

inline void test_to_vector_after_transpose(Backend& be) {
    std::cout << "[Factories 10] to_vector: non-contiguous (transposed) reads correctly\n";
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

inline void test_to_vector_scalar(Backend& be) {
    std::cout << "[Factories 11] to_vector: scalar {1} returns single-element vector\n";
    Tensor t = Tensor::from_data<double>({42.0}, {1}, be);
    auto v   = t.to_vector<double>();
    CHECK(v.size() == 1);
    CHECK_NEAR(v[0], 42.0, 1e-12);
}

// ── has_nan / has_inf ────────────────────────────────────────────────────────

inline void test_has_nan_false(Backend& be) {
    std::cout << "[Factories 12] has_nan: returns false for clean tensor\n";
    Tensor t = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be);
    CHECK(!has_nan(t));
}

inline void test_has_nan_true(Backend& be) {
    std::cout << "[Factories 13] has_nan: returns true when any element is nan\n";
    Tensor t = Tensor::from_data<double>({1.0, -1.0, 3.0}, {3}, be);
    Tensor r = t.log();
    CHECK(has_nan(r));  // log(-1) = nan
}

inline void test_has_inf_false(Backend& be) {
    std::cout << "[Factories 14] has_inf: returns false for clean tensor\n";
    Tensor t = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be);
    CHECK(!has_inf(t));
}

inline void test_has_inf_true(Backend& be) {
    std::cout << "[Factories 15] has_inf: returns true when any element is ±inf\n";
    Tensor a = Tensor::from_data<double>({1.0, 2.0}, {2}, be);
    Tensor b = Tensor::from_data<double>({0.0, 1.0}, {2}, be);
    Tensor r = a.div(b);
    CHECK(has_inf(r));  // r[0] = +inf
}

// ── max_abs_diff ─────────────────────────────────────────────────────────────

inline void test_max_abs_diff_zero(Backend& be) {
    std::cout << "[Factories 16] max_abs_diff: identical tensors → 0.0\n";
    Tensor a = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be);
    Tensor b = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be);
    CHECK_NEAR(max_abs_diff(a, b), 0.0, 1e-15);
}

inline void test_max_abs_diff_value(Backend& be) {
    std::cout << "[Factories 17] max_abs_diff: correct maximum absolute difference\n";
    // differences: |1-1|=0, |4-2|=2, |3-6|=3 → max = 3
    Tensor a = Tensor::from_data<double>({1.0, 4.0, 3.0}, {3}, be);
    Tensor b = Tensor::from_data<double>({1.0, 2.0, 6.0}, {3}, be);
    CHECK_NEAR(max_abs_diff(a, b), 3.0, 1e-12);
}

inline void test_max_abs_diff_shape_mismatch_throws(Backend& be) {
    std::cout << "[Factories 18] max_abs_diff: shape mismatch throws\n";
    Tensor a = Tensor::from_data<double>({1.0, 2.0}, {2}, be);
    Tensor b = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be);
    bool threw = false;
    try { max_abs_diff(a, b); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

// ── shape_str / dtype_str ────────────────────────────────────────────────────

inline void test_shape_str(Backend& be) {
    std::cout << "[Factories 19] shape_str: correct string for {2,3} tensor\n";
    Tensor t = Tensor::zeros({2, 3}, be);
    CHECK(shape_str(t) == "[2, 3]");
}

inline void test_dtype_str(Backend& be) {
    std::cout << "[Factories 20] dtype_str: Float64 tensor returns \"Float64\"\n";
    Tensor t = Tensor::zeros({2}, be);
    CHECK(dtype_str(t) == "Float64");
}

// ── memory ───────────────────────────────────────────────────────────────────

inline void test_factories_memory_clean(Backend& be) {
    std::cout << "[Factories 21] memory: bytes_allocated == 0 after all factory tensors destruct\n";
    {
        Tensor a = Tensor::ones({3, 4}, be);
        Tensor b = Tensor::full({2, 2}, 5.0, be);
        Tensor c = Tensor::zeros_like(a);
        Tensor d = Tensor::ones_like(b);
        auto   v = a.to_vector<double>();
        (void)v;
    }
    CHECK(be.memory_manager()->bytes_allocated() == 0);
}

// ── run all ──────────────────────────────────────────────────────────────────

inline void run_shared_factories(Backend& be) {
    test_ones_values(be);
    test_ones_requires_grad(be);
    test_full_values(be);
    test_full_negative_value(be);
    test_zeros_like_shape_dtype_backend(be);
    test_zeros_like_independent_buffer(be);
    test_ones_like_values(be);
    test_ones_like_requires_grad(be);
    test_to_vector_contiguous(be);
    test_to_vector_after_transpose(be);
    test_to_vector_scalar(be);
    test_has_nan_false(be);
    test_has_nan_true(be);
    test_has_inf_false(be);
    test_has_inf_true(be);
    test_max_abs_diff_zero(be);
    test_max_abs_diff_value(be);
    test_max_abs_diff_shape_mismatch_throws(be);
    test_shape_str(be);
    test_dtype_str(be);
    test_factories_memory_clean(be);
}

} // namespace otter::test::shared
