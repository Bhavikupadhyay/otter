#include "test_utils.h"

#include <cstddef>
#include <vector>

#include "otter/tensor.h"
#include "otter/autograd/no_grad_guard.h"
#include "otter/backends/cpu.h"

// Tests for Tensor autograd fields, GradAccumulator, NoGradGuard.
// All expected values are derived from the documented API contracts.
// bytes_allocated() is not checked here (requires CPUMemoryManager access);
// memory hygiene is covered in test_memory.cpp.

namespace otter::test {

// ── Tensor::zeros — requires_grad ────────────────────────────────────────────

void test_tensor_zeros_default_no_grad() {
    std::cout << "[Tensor 1] zeros: default — leaf, requires_grad=false, grad undefined\n";
    Backend& be = cpu_backend();
    Tensor t = Tensor::zeros({2, 3}, be);
    CHECK(!t.requires_grad());
    CHECK(t.is_leaf());
    CHECK(!t.grad().defined());
}

void test_tensor_zeros_with_grad() {
    std::cout << "[Tensor 2] zeros: requires_grad=true — leaf, requires_grad=true, grad undefined\n";
    Backend& be = cpu_backend();
    Tensor t = Tensor::zeros({3}, be, DType::Float64, /*requires_grad=*/true);
    CHECK(t.requires_grad());
    CHECK(t.is_leaf());
    CHECK(!t.grad().defined()); // no backward yet — grad accumulator empty
}

// ── Tensor::from_data — requires_grad ────────────────────────────────────────

void test_tensor_from_data_with_grad() {
    std::cout << "[Tensor 3] from_data: requires_grad=true — correct values, grad=false initially\n";
    Backend& be = cpu_backend();
    Tensor t = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be, /*requires_grad=*/true);
    CHECK(t.requires_grad());
    CHECK(t.is_leaf());
    CHECK_NEAR(t.at({0}), 1.0, 1e-12);
    CHECK_NEAR(t.at({1}), 2.0, 1e-12);
    CHECK_NEAR(t.at({2}), 3.0, 1e-12);
    CHECK(!t.grad().defined());
}

// ── Tensor::zero_grad ─────────────────────────────────────────────────────────

void test_tensor_zero_grad_noop_when_no_accum() {
    std::cout << "[Tensor 4] zero_grad: no-op when grad not yet accumulated\n";
    Backend& be = cpu_backend();
    Tensor t = Tensor::zeros({2}, be, DType::Float64, true);
    t.zero_grad(); // must not crash
    CHECK(!t.grad().defined());
}

// ── Tensor::detach ────────────────────────────────────────────────────────────

void test_tensor_detach_clears_grad_fields() {
    std::cout << "[Tensor 5] detach: requires_grad=false, grad undefined, same shape\n";
    Backend& be = cpu_backend();
    Tensor t = Tensor::zeros({2, 3}, be, DType::Float64, true);
    Tensor d = t.detach();

    CHECK(!d.requires_grad());
    CHECK(!d.grad().defined());
    // Shape and values preserved (shared buffer — no copy)
    CHECK(d.shape() == t.shape());
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            CHECK_NEAR(d.at({r, c}), 0.0, 1e-12);
}

void test_tensor_detach_shares_buffer_values() {
    std::cout << "[Tensor 6] detach: shares buffer — same values, independent autograd state\n";
    Backend& be = cpu_backend();
    Tensor t = Tensor::from_data<double>({1.0, 2.0, 3.0, 4.0}, {2, 2}, be, true);
    Tensor d = t.detach();

    // Shared buffer: same values
    CHECK_NEAR(d.at({0, 0}), 1.0, 1e-12);
    CHECK_NEAR(d.at({0, 1}), 2.0, 1e-12);
    CHECK_NEAR(d.at({1, 0}), 3.0, 1e-12);
    CHECK_NEAR(d.at({1, 1}), 4.0, 1e-12);

    // Independent autograd state
    CHECK(t.requires_grad());
    CHECK(!d.requires_grad());
}

void test_tensor_detach_leaf_of_computed_stays_non_leaf() {
    // detach() does NOT change is_leaf_. A leaf tensor detaches to another leaf.
    // (A computed tensor would detach without changing its is_leaf_=false — but we
    // can't test that until Operations exist in step 3.)
    std::cout << "[Tensor 7] detach: leaf tensor detaches to leaf\n";
    Backend& be = cpu_backend();
    Tensor t = Tensor::zeros({2}, be, DType::Float64, true);
    Tensor d = t.detach();
    // Both are leaves (t was user-created; detach doesn't force is_leaf_=false here)
    CHECK(t.is_leaf());
    CHECK(d.is_leaf()); // user-created leaf detaches to non-grad leaf — still technically leaf
}

// ── NoGradGuard ───────────────────────────────────────────────────────────────

void test_no_grad_guard_default_on() {
    std::cout << "[Tensor 8] NoGradGuard: grad_mode defaults to true\n";
    CHECK(NoGradGuard::grad_mode());
}

void test_no_grad_guard_disables_and_restores() {
    std::cout << "[Tensor 9] NoGradGuard: disables grad tracking, restores on destruction\n";
    CHECK(NoGradGuard::grad_mode()); // pre-condition
    {
        NoGradGuard ng;
        CHECK(!NoGradGuard::grad_mode()); // disabled inside scope
    }
    CHECK(NoGradGuard::grad_mode()); // restored after scope
}

void test_no_grad_guard_nested_safe() {
    std::cout << "[Tensor 10] NoGradGuard: nested guards restore correctly\n";
    CHECK(NoGradGuard::grad_mode()); // pre-condition
    {
        NoGradGuard ng1;
        CHECK(!NoGradGuard::grad_mode()); // outer guard disables
        {
            NoGradGuard ng2;
            CHECK(!NoGradGuard::grad_mode()); // inner guard: already false
        }
        // ng2 restores to what it saved (false from ng1), so still false
        CHECK(!NoGradGuard::grad_mode());
    }
    // ng1 restores to what it saved (true), so true again
    CHECK(NoGradGuard::grad_mode());
}

// ── Tensor::is_unique ────────────────────────────────────────────────────────

void test_is_unique_single_tensor() {
    std::cout << "[Tensor 11] is_unique: freshly allocated tensor is sole buffer holder\n";
    Backend& be = cpu_backend();
    Tensor t = Tensor::zeros({4}, be);
    CHECK(t.is_unique());
}

void test_is_unique_false_after_copy() {
    std::cout << "[Tensor 12] is_unique: copy shares buffer — both return false\n";
    Backend& be = cpu_backend();
    Tensor t = Tensor::zeros({4}, be);
    Tensor u = t;  // value-type copy shares the same Buffer
    CHECK(!t.is_unique());
    CHECK(!u.is_unique());
}

void test_is_unique_restored_when_copy_destroyed() {
    std::cout << "[Tensor 13] is_unique: true again after the only other holder is destroyed\n";
    Backend& be = cpu_backend();
    Tensor t = Tensor::zeros({4}, be);
    {
        Tensor u = t;
        CHECK(!t.is_unique());  // two holders inside the block
    }
    CHECK(t.is_unique());  // u is gone; t is the sole holder again
}

void test_is_unique_false_for_view() {
    std::cout << "[Tensor 14] is_unique: reshape view shares buffer — original returns false\n";
    Backend& be = cpu_backend();
    Tensor t = Tensor::zeros({2, 3}, be);
    Tensor v = t.reshape({6});
    // Both share the same Buffer, so neither is unique.
    CHECK(!t.is_unique());
    CHECK(!v.is_unique());
}

// ── Run all ───────────────────────────────────────────────────────────────────

void run_tensor_tests() {
    test_tensor_zeros_default_no_grad();
    test_tensor_zeros_with_grad();
    test_tensor_from_data_with_grad();
    test_tensor_zero_grad_noop_when_no_accum();
    test_tensor_detach_clears_grad_fields();
    test_tensor_detach_shares_buffer_values();
    test_tensor_detach_leaf_of_computed_stays_non_leaf();
    test_no_grad_guard_default_on();
    test_no_grad_guard_disables_and_restores();
    test_no_grad_guard_nested_safe();
    test_is_unique_single_tensor();
    test_is_unique_false_after_copy();
    test_is_unique_restored_when_copy_destroyed();
    test_is_unique_false_for_view();
}

} // namespace otter::test
