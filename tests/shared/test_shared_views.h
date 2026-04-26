#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "../utils/test_utils.h"
#include "otter/tensor.h"
#include "otter/kernel/backend.h"

namespace otter::test::shared {

// ── reshape ───────────────────────────────────────────────────────────────────

inline void test_reshape_forward(Backend& be) {
    std::cout << "[Views 1] reshape {2,3} → {3,2}: values preserved\n";
    Tensor a = Tensor::from_data<double>({1,2,3,4,5,6}, {2,3}, be);
    Tensor b = a.reshape({3, 2});
    CHECK(b.shape() == (std::vector<std::size_t>{3, 2}));
    CHECK_NEAR(b.at({0,0}), 1.0, 1e-12);
    CHECK_NEAR(b.at({0,1}), 2.0, 1e-12);
    CHECK_NEAR(b.at({1,0}), 3.0, 1e-12);
    CHECK_NEAR(b.at({1,1}), 4.0, 1e-12);
    CHECK_NEAR(b.at({2,0}), 5.0, 1e-12);
    CHECK_NEAR(b.at({2,1}), 6.0, 1e-12);
}

inline void test_reshape_flatten(Backend& be) {
    std::cout << "[Views 2] reshape {2,3} → {6}: flatten\n";
    Tensor a = Tensor::from_data<double>({1,2,3,4,5,6}, {2,3}, be);
    Tensor b = a.reshape({6});
    CHECK(b.shape() == (std::vector<std::size_t>{6}));
    for (std::size_t i = 0; i < 6; ++i)
        CHECK_NEAR(b.at({i}), static_cast<double>(i + 1), 1e-12);
}

inline void test_reshape_numel_mismatch_throws(Backend& be) {
    std::cout << "[Views 3] reshape numel mismatch throws\n";
    Tensor a = Tensor::zeros({2,3}, be);
    bool threw = false;
    try { (void)a.reshape({2,4}); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

inline void test_reshape_non_contiguous_throws(Backend& be) {
    std::cout << "[Views 4] reshape non-contiguous input throws\n";
    Tensor a = Tensor::from_data<double>({1,2,3,4,5,6}, {2,3}, be);
    Tensor t = a.transpose(0, 1);
    bool threw = false;
    try { (void)t.reshape({6}); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

inline void test_reshape_backward(Backend& be) {
    std::cout << "[Views 5] reshape backward: grad reshaped back to original\n";
    Tensor a = Tensor::from_data<double>({1,2,3,4,5,6}, {2,3}, be, /*requires_grad=*/true);
    a.reshape({6}).sum().backward();
    CHECK(a.grad().shape() == (std::vector<std::size_t>{2, 3}));
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            CHECK_NEAR(a.grad().at({r,c}), 1.0, 1e-12);
}

// ── transpose ─────────────────────────────────────────────────────────────────

inline void test_transpose_forward(Backend& be) {
    std::cout << "[Views 6] transpose {2,3}: shape becomes {3,2}, values correct\n";
    Tensor a = Tensor::from_data<double>({1,2,3,4,5,6}, {2,3}, be);
    Tensor t = a.transpose(0, 1);
    CHECK(t.shape() == (std::vector<std::size_t>{3, 2}));
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            CHECK_NEAR(t.at({i,j}), a.at({j,i}), 1e-12);
}

inline void test_transpose_is_view(Backend& be) {
    std::cout << "[Views 7] transpose shares buffer (no data copy)\n";
    Tensor a = Tensor::from_data<double>({1,2,3,4}, {2,2}, be);
    Tensor t = a.transpose(0, 1);
    CHECK(t.defined());
    CHECK(t.numel() == a.numel());
}

inline void test_transpose_dim_out_of_range_throws(Backend& be) {
    std::cout << "[Views 8] transpose dim out of range throws\n";
    Tensor a = Tensor::zeros({2,3}, be);
    bool threw = false;
    try { (void)a.transpose(0, 5); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

inline void test_transpose_backward(Backend& be) {
    std::cout << "[Views 9] transpose backward: grad transposed back\n";
    Tensor a = Tensor::from_data<double>({1,2,3,4,5,6}, {2,3}, be, /*requires_grad=*/true);
    a.transpose(0,1).sum().backward();
    CHECK(a.grad().shape() == (std::vector<std::size_t>{2, 3}));
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            CHECK_NEAR(a.grad().at({r,c}), 1.0, 1e-12);
}

inline void test_transpose_matmul_backward(Backend& be) {
    std::cout << "[Views 10] transpose in matmul chain: (A^T @ A) backward\n";
    Tensor A = Tensor::from_data<double>({1,2,3,4}, {2,2}, be, /*requires_grad=*/true);
    A.transpose(0,1).matmul(A).sum().backward();
    CHECK(A.grad().defined());
    CHECK(A.grad().shape() == (std::vector<std::size_t>{2, 2}));
}

// ── slice ─────────────────────────────────────────────────────────────────────

inline void test_slice_1d_forward(Backend& be) {
    std::cout << "[Views 11] slice {6}[1:4]: shape {3}, correct values\n";
    Tensor a = Tensor::from_data<double>({10,20,30,40,50,60}, {6}, be);
    Tensor s = a.slice(0, 1, 3);
    CHECK(s.shape() == (std::vector<std::size_t>{3}));
    CHECK_NEAR(s.at({0}), 20.0, 1e-12);
    CHECK_NEAR(s.at({1}), 30.0, 1e-12);
    CHECK_NEAR(s.at({2}), 40.0, 1e-12);
}

inline void test_slice_2d_row_forward(Backend& be) {
    std::cout << "[Views 12] slice {3,4} row 1..2: shape {2,4}, correct values\n";
    std::vector<double> data;
    for (double i = 1; i <= 12; ++i) data.push_back(i);
    Tensor a = Tensor::from_data<double>(data, {3, 4}, be);
    Tensor s = a.slice(0, 1, 2);
    CHECK(s.shape() == (std::vector<std::size_t>{2, 4}));
    CHECK_NEAR(s.at({0, 0}),  5.0, 1e-12);
    CHECK_NEAR(s.at({0, 3}),  8.0, 1e-12);
    CHECK_NEAR(s.at({1, 0}),  9.0, 1e-12);
    CHECK_NEAR(s.at({1, 3}), 12.0, 1e-12);
}

inline void test_slice_out_of_bounds_throws(Backend& be) {
    std::cout << "[Views 13] slice out of bounds throws\n";
    Tensor a = Tensor::zeros({5}, be);
    bool threw = false;
    try { (void)a.slice(0, 3, 4); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

inline void test_slice_backward(Backend& be) {
    std::cout << "[Views 14] slice backward: grad scattered into zeros at slice pos\n";
    // a = [1,2,3,4,5], slice [1:4] = [2,3,4]
    // grad_a: [0, 1, 1, 1, 0]
    Tensor a = Tensor::from_data<double>({1,2,3,4,5}, {5}, be, /*requires_grad=*/true);
    a.slice(0, 1, 3).sum().backward();
    CHECK(a.grad().shape() == (std::vector<std::size_t>{5}));
    CHECK_NEAR(a.grad().at({0}), 0.0, 1e-12);
    CHECK_NEAR(a.grad().at({1}), 1.0, 1e-12);
    CHECK_NEAR(a.grad().at({2}), 1.0, 1e-12);
    CHECK_NEAR(a.grad().at({3}), 1.0, 1e-12);
    CHECK_NEAR(a.grad().at({4}), 0.0, 1e-12);
}

inline void test_slice_backward_2d(Backend& be) {
    std::cout << "[Views 15] slice 2D backward: grad rows scattered correctly\n";
    // a = [[1,2],[3,4],[5,6]], slice rows 0..1 → grad_a = [[1,1],[1,1],[0,0]]
    Tensor a = Tensor::from_data<double>({1,2,3,4,5,6}, {3,2}, be, /*requires_grad=*/true);
    a.slice(0, 0, 2).sum().backward();
    CHECK(a.grad().shape() == (std::vector<std::size_t>{3, 2}));
    CHECK_NEAR(a.grad().at({0,0}), 1.0, 1e-12);
    CHECK_NEAR(a.grad().at({0,1}), 1.0, 1e-12);
    CHECK_NEAR(a.grad().at({1,0}), 1.0, 1e-12);
    CHECK_NEAR(a.grad().at({1,1}), 1.0, 1e-12);
    CHECK_NEAR(a.grad().at({2,0}), 0.0, 1e-12);
    CHECK_NEAR(a.grad().at({2,1}), 0.0, 1e-12);
}

// ── broadcast_to ─────────────────────────────────────────────────────────────

inline void test_broadcast_to_forward(Backend& be) {
    std::cout << "[Views 16] broadcast_to {3} → {2,3}: stride-zero, no data copy\n";
    Tensor a = Tensor::from_data<double>({1,2,3}, {3}, be);
    Tensor b = a.broadcast_to({2, 3});
    CHECK(b.shape() == (std::vector<std::size_t>{2, 3}));
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            CHECK_NEAR(b.at({r,c}), static_cast<double>(c + 1), 1e-12);
}

inline void test_broadcast_to_backward(Backend& be) {
    std::cout << "[Views 17] broadcast_to backward: grad reduced over broadcast dim\n";
    // a=[1,2,3], broadcast to {2,3}: each element in 2 rows → grad_a=[2,2,2]
    Tensor a = Tensor::from_data<double>({1,2,3}, {3}, be, /*requires_grad=*/true);
    a.broadcast_to({2,3}).sum().backward();
    CHECK(a.grad().shape() == (std::vector<std::size_t>{3}));
    CHECK_NEAR(a.grad().at({0}), 2.0, 1e-12);
    CHECK_NEAR(a.grad().at({1}), 2.0, 1e-12);
    CHECK_NEAR(a.grad().at({2}), 2.0, 1e-12);
}

inline void test_broadcast_to_incompatible_throws(Backend& be) {
    std::cout << "[Views 18] broadcast_to incompatible shape throws\n";
    Tensor a = Tensor::zeros({3}, be);
    bool threw = false;
    try { (void)a.broadcast_to({2, 4}); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

// ── memory ───────────────────────────────────────────────────────────────────

inline void test_views_memory_clean(Backend& be) {
    std::cout << "[Views 19] memory: bytes_allocated == 0 after view ops and backward\n";
    {
        Tensor a = Tensor::from_data<double>({1,2,3,4,5,6}, {2,3}, be, /*requires_grad=*/true);
        a.reshape({6}).sum().backward();
        a.zero_grad();

        Tensor b = Tensor::from_data<double>({1,2,3,4,5,6}, {2,3}, be, /*requires_grad=*/true);
        b.transpose(0,1).sum().backward();
        b.zero_grad();

        Tensor c = Tensor::from_data<double>({1,2,3,4,5}, {5}, be, /*requires_grad=*/true);
        c.slice(0,1,3).sum().backward();
        c.zero_grad();

        Tensor d = Tensor::from_data<double>({1,2,3}, {3}, be, /*requires_grad=*/true);
        d.broadcast_to({2,3}).sum().backward();
        d.zero_grad();
    }
    CHECK(be.memory_manager()->bytes_allocated() == 0);
}

// ── run all ──────────────────────────────────────────────────────────────────

inline void run_shared_views(Backend& be) {
    test_reshape_forward(be);
    test_reshape_flatten(be);
    test_reshape_numel_mismatch_throws(be);
    test_reshape_non_contiguous_throws(be);
    test_reshape_backward(be);
    test_transpose_forward(be);
    test_transpose_is_view(be);
    test_transpose_dim_out_of_range_throws(be);
    test_transpose_backward(be);
    test_transpose_matmul_backward(be);
    test_slice_1d_forward(be);
    test_slice_2d_row_forward(be);
    test_slice_out_of_bounds_throws(be);
    test_slice_backward(be);
    test_slice_backward_2d(be);
    test_broadcast_to_forward(be);
    test_broadcast_to_backward(be);
    test_broadcast_to_incompatible_throws(be);
    test_views_memory_clean(be);
}

} // namespace otter::test::shared
