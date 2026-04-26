#pragma once

#include <cstddef>
#include <vector>

#include "../utils/test_utils.h"
#include "otter/tensor.h"
#include "otter/autograd/no_grad_guard.h"
#include "otter/kernel/backend.h"

namespace otter::test::shared {

// ── Tensor::zeros — requires_grad ────────────────────────────────────────────

inline void test_tensor_zeros_default_no_grad(Backend& be) {
    std::cout << "[Tensor 1] zeros: default — leaf, requires_grad=false, grad undefined\n";
    Tensor t = Tensor::zeros({2, 3}, be);
    CHECK(!t.requires_grad());
    CHECK(t.is_leaf());
    CHECK(!t.grad().defined());
}

inline void test_tensor_zeros_with_grad(Backend& be) {
    std::cout << "[Tensor 2] zeros: requires_grad=true — leaf, requires_grad=true, grad undefined\n";
    Tensor t = Tensor::zeros({3}, be, DType::Float64, /*requires_grad=*/true);
    CHECK(t.requires_grad());
    CHECK(t.is_leaf());
    CHECK(!t.grad().defined());
}

// ── Tensor::from_data — requires_grad ────────────────────────────────────────

inline void test_tensor_from_data_with_grad(Backend& be) {
    std::cout << "[Tensor 3] from_data: requires_grad=true — correct values, grad undefined\n";
    Tensor t = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be, /*requires_grad=*/true);
    CHECK(t.requires_grad());
    CHECK(t.is_leaf());
    CHECK_NEAR(t.at({0}), 1.0, 1e-12);
    CHECK_NEAR(t.at({1}), 2.0, 1e-12);
    CHECK_NEAR(t.at({2}), 3.0, 1e-12);
    CHECK(!t.grad().defined());
}

// ── Tensor::zero_grad ─────────────────────────────────────────────────────────

inline void test_tensor_zero_grad_noop_when_no_accum(Backend& be) {
    std::cout << "[Tensor 4] zero_grad: no-op when grad not yet accumulated\n";
    Tensor t = Tensor::zeros({2}, be, DType::Float64, true);
    t.zero_grad();  // must not crash
    CHECK(!t.grad().defined());
}

// ── Tensor::detach ────────────────────────────────────────────────────────────

inline void test_tensor_detach_clears_grad_fields(Backend& be) {
    std::cout << "[Tensor 5] detach: requires_grad=false, grad undefined, same shape\n";
    Tensor t = Tensor::zeros({2, 3}, be, DType::Float64, true);
    Tensor d = t.detach();
    CHECK(!d.requires_grad());
    CHECK(!d.grad().defined());
    CHECK(d.shape() == t.shape());
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            CHECK_NEAR(d.at({r, c}), 0.0, 1e-12);
}

inline void test_tensor_detach_shares_buffer_values(Backend& be) {
    std::cout << "[Tensor 6] detach: shares buffer — same values, independent autograd\n";
    Tensor t = Tensor::from_data<double>({1.0, 2.0, 3.0, 4.0}, {2, 2}, be, true);
    Tensor d = t.detach();
    CHECK_NEAR(d.at({0, 0}), 1.0, 1e-12);
    CHECK_NEAR(d.at({0, 1}), 2.0, 1e-12);
    CHECK_NEAR(d.at({1, 0}), 3.0, 1e-12);
    CHECK_NEAR(d.at({1, 1}), 4.0, 1e-12);
    CHECK(t.requires_grad());
    CHECK(!d.requires_grad());
}

inline void test_tensor_detach_leaf_stays_leaf(Backend& be) {
    std::cout << "[Tensor 7] detach: leaf tensor detaches to leaf\n";
    Tensor t = Tensor::zeros({2}, be, DType::Float64, true);
    Tensor d = t.detach();
    CHECK(t.is_leaf());
    CHECK(d.is_leaf());
}

// ── NoGradGuard ───────────────────────────────────────────────────────────────

inline void test_no_grad_guard_default_on(Backend& /*be*/) {
    std::cout << "[Tensor 8] NoGradGuard: grad_mode defaults to true\n";
    CHECK(NoGradGuard::grad_mode());
}

inline void test_no_grad_guard_disables_and_restores(Backend& /*be*/) {
    std::cout << "[Tensor 9] NoGradGuard: disables grad tracking, restores on destruction\n";
    CHECK(NoGradGuard::grad_mode());
    {
        NoGradGuard ng;
        CHECK(!NoGradGuard::grad_mode());
    }
    CHECK(NoGradGuard::grad_mode());
}

inline void test_no_grad_guard_nested_safe(Backend& /*be*/) {
    std::cout << "[Tensor 10] NoGradGuard: nested guards restore correctly\n";
    CHECK(NoGradGuard::grad_mode());
    {
        NoGradGuard ng1;
        CHECK(!NoGradGuard::grad_mode());
        {
            NoGradGuard ng2;
            CHECK(!NoGradGuard::grad_mode());
        }
        CHECK(!NoGradGuard::grad_mode());  // ng2 restored to false (set by ng1)
    }
    CHECK(NoGradGuard::grad_mode());  // ng1 restored to true
}

// ── Tensor::is_unique ────────────────────────────────────────────────────────

inline void test_is_unique_single_tensor(Backend& be) {
    std::cout << "[Tensor 11] is_unique: fresh tensor is sole buffer holder\n";
    Tensor t = Tensor::zeros({4}, be);
    CHECK(t.is_unique());
}

inline void test_is_unique_false_after_copy(Backend& be) {
    std::cout << "[Tensor 12] is_unique: copy shares buffer — both return false\n";
    Tensor t = Tensor::zeros({4}, be);
    Tensor u = t;
    CHECK(!t.is_unique());
    CHECK(!u.is_unique());
}

inline void test_is_unique_restored_when_copy_destroyed(Backend& be) {
    std::cout << "[Tensor 13] is_unique: true again after the only other holder is destroyed\n";
    Tensor t = Tensor::zeros({4}, be);
    {
        Tensor u = t;
        CHECK(!t.is_unique());
    }
    CHECK(t.is_unique());
}

inline void test_is_unique_false_for_view(Backend& be) {
    std::cout << "[Tensor 14] is_unique: reshape view shares buffer — original returns false\n";
    Tensor t = Tensor::zeros({2, 3}, be);
    Tensor v = t.reshape({6});
    CHECK(!t.is_unique());
    CHECK(!v.is_unique());
}

// ── run all ──────────────────────────────────────────────────────────────────

inline void run_shared_tensor(Backend& be) {
    test_tensor_zeros_default_no_grad(be);
    test_tensor_zeros_with_grad(be);
    test_tensor_from_data_with_grad(be);
    test_tensor_zero_grad_noop_when_no_accum(be);
    test_tensor_detach_clears_grad_fields(be);
    test_tensor_detach_shares_buffer_values(be);
    test_tensor_detach_leaf_stays_leaf(be);
    test_no_grad_guard_default_on(be);
    test_no_grad_guard_disables_and_restores(be);
    test_no_grad_guard_nested_safe(be);
    test_is_unique_single_tensor(be);
    test_is_unique_false_after_copy(be);
    test_is_unique_restored_when_copy_destroyed(be);
    test_is_unique_false_for_view(be);
}

} // namespace otter::test::shared
