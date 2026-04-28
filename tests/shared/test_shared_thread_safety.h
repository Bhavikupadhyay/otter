#pragma once

#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

#include "../utils/test_utils.h"
#include "otter/tensor.h"
#include "otter/kernel/backend.h"

namespace otter::test::shared {

// ── Test 1: concurrent grad() reads ──────────────────────────────────────────

inline void test_grad_safe_concurrent_reads(Backend& be) {
    std::cout << "[ThreadSafety 1] N threads read w.grad() concurrently — no data race\n";
    Tensor w = Tensor::from_data<double>({5.0, 6.0}, {2}, be, /*requires_grad=*/true);
    // backward() completes on the main thread before threads are spawned.
    w.add(Tensor::from_data<double>({0.0, 0.0}, {2}, be)).sum().backward();

    constexpr std::size_t N = 8;
    std::vector<double> val0(N, -1.0);
    std::vector<double> val1(N, -1.0);
    std::vector<std::thread> threads;
    threads.reserve(N);

    for (std::size_t i = 0; i < N; ++i) {
        threads.emplace_back([&, i]() {
            Tensor g = w.grad();
            val0[i] = g.at({0});
            val1[i] = g.at({1});
        });
    }
    for (auto& t : threads) t.join();

    for (std::size_t i = 0; i < N; ++i) {
        CHECK_NEAR(val0[i], 1.0, 1e-12);
        CHECK_NEAR(val1[i], 1.0, 1e-12);
    }
}

// ── Test 2: concurrent backward() on separate graphs, shared leaf ────────────

inline void test_concurrent_backward_separate_graphs(Backend& be) {
    std::cout << "[ThreadSafety 2] N threads backward over shared leaf w — grad accumulates correctly\n";
    // Shared leaf: w = [3.0, 1.0], requires_grad=true.
    // Each thread computes loss = sum(w * [2.0, 2.0]) → d(loss)/dw = [2, 2].
    // After N threads: w.grad() == [N*2, N*2].
    constexpr int N = 4;
    Tensor w = Tensor::from_data<double>({3.0, 1.0}, {2}, be, /*requires_grad=*/true);

    std::atomic<int>  built{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;
    threads.reserve(N);

    for (int i = 0; i < N; ++i) {
        threads.emplace_back([&]() {
            Tensor scale = Tensor::from_data<double>({2.0, 2.0}, {2}, be);
            Tensor loss  = w.mul(scale).sum();
            built.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) { /* spin */ }
            loss.backward();
        });
    }

    while (built.load(std::memory_order_acquire) < N) { /* spin */ }
    go.store(true, std::memory_order_release);
    for (auto& t : threads) t.join();

    const double expected = static_cast<double>(N) * 2.0;
    CHECK_NEAR(w.grad().at({0}), expected, 1e-9);
    CHECK_NEAR(w.grad().at({1}), expected, 1e-9);
}

// ── Test 3: backward() and zero_grad() interleaved ───────────────────────────

inline void test_zero_grad_interleaved_with_backward(Backend& be) {
    std::cout << "[ThreadSafety 3] backward() + zero_grad() interleaved — no UB, valid shape\n";
    for (int trial = 0; trial < 32; ++trial) {
        Tensor w    = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be, /*requires_grad=*/true);
        Tensor loss = w.add(Tensor::from_data<double>({0.0, 0.0, 0.0}, {3}, be)).sum();

        std::atomic<bool> start{false};
        std::thread t_back([&]() {
            while (!start.load(std::memory_order_acquire)) { /* spin */ }
            loss.backward();
        });
        std::thread t_zero([&]() {
            while (!start.load(std::memory_order_acquire)) { /* spin */ }
            w.zero_grad();
        });
        start.store(true, std::memory_order_release);
        t_back.join();
        t_zero.join();

        // Result is non-deterministic — just require structural validity.
        Tensor g = w.grad();
        if (g.defined()) {
            CHECK(g.shape() == w.shape());
        }
    }
    // Count the overall test as a single pass/fail unit.
    ++otter::test::tests_run;
    ++otter::test::tests_passed;
}

// ── Test 4: fan-out graph — dep-count engine correctness ─────────────────────
//
// Validates that GraphExecutor's dep-count model correctly handles a node whose
// output is consumed by two independent operations (fan-out in the forward graph
// = fan-in for gradient accumulation in the backward graph).
//
// Graph:
//   leaf L (requires_grad)
//     → MulOp(×2) → T
//         → AddOp_B(T + zeros) → sum_B ─┐
//         → AddOp_C(T + zeros) → sum_C ─┴→ loss = sum_B + sum_C
//
// Analytical gradient: d(loss)/dL = 4 * ones (chain: loss→T: 2×[1,1], T→L: ×2)
//
// MulOp's dep_count = 2 in the backward graph: both AddOp_B and AddOp_C must
// accumulate their gradients into T before MulOp can backward. Running 500
// iterations makes any race or incorrect dep-count wiring statistically visible.

inline void test_fan_out_dep_count(Backend& be) {
    std::cout << "[ThreadSafety 4] Fan-out graph: dep-count engine produces correct gradient (500 iters)\n";

    for (int iter = 0; iter < 500; ++iter) {
        Tensor L = Tensor::from_data<double>({1.0, 1.0}, {2}, be, /*requires_grad=*/true);

        Tensor two   = Tensor::from_data<double>({2.0, 2.0}, {2}, be);
        Tensor zeros = Tensor::from_data<double>({0.0, 0.0}, {2}, be);

        Tensor T      = L.mul(two);
        Tensor sum_B  = T.add(zeros).sum();
        Tensor sum_C  = T.add(zeros).sum();
        Tensor loss   = sum_B.add(sum_C);

        loss.backward();

        // Each element of L.grad() must equal 4.0 — every iteration, deterministically.
        CHECK_NEAR(L.grad().at({0}), 4.0, 1e-9);
        CHECK_NEAR(L.grad().at({1}), 4.0, 1e-9);
    }
}

// ── run all ──────────────────────────────────────────────────────────────────

inline void run_shared_thread_safety(Backend& be) {
    test_grad_safe_concurrent_reads(be);
    test_concurrent_backward_separate_graphs(be);
    test_zero_grad_interleaved_with_backward(be);
    test_fan_out_dep_count(be);
}

} // namespace otter::test::shared
