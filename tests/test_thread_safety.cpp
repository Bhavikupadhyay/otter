#include "test_utils.h"

#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

#include "otter/backends/cpu.h"
#include "otter/tensor.h"

// Thread-safety tests for Tensor::grad(), Tensor::backward(), and GradAccumulator.
//
// Scenarios covered:
//   1. Concurrent grad() reads after backward() completes.
//   2. Concurrent backward() on SEPARATE graphs sharing a leaf weight (data-parallel
//      training). Each thread owns its own intermediate Operations; only the leaf's
//      GradAccumulator is shared.
//   3. backward() and zero_grad() interleaved: result is non-deterministic but must
//      never cause undefined behaviour, corrupt memory, or produce an invalid shape.
//
// Note: concurrent backward() on graphs sharing INTERMEDIATE Operations with
// retain_graph=false is not tested here — that scenario is explicitly out of scope
// (see the Thread Safety Scope Statement in the sprint plan).

namespace otter::test {

// ── Test 1: concurrent grad() reads ──────────────────────────────────────────

void test_grad_safe_concurrent_reads() {
    std::cout << "[ThreadSafety 1] N threads read w.grad() concurrently — no data race\n";
    Backend& be = cpu_backend();

    Tensor w = Tensor::from_data<double>({5.0, 6.0}, {2}, be, /*requires_grad=*/true);
    // backward() completes on the main thread before threads are spawned.
    // grad(w) should be [1, 1] from the sum.
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

void test_concurrent_backward_separate_graphs() {
    std::cout << "[ThreadSafety 2] N threads backward over shared leaf w — grad accumulates correctly\n";
    Backend& be = cpu_backend();

    // Shared leaf: w = [3.0, 1.0], requires_grad=true.
    // Each thread computes loss = sum(w * [2.0, 2.0]).
    // d(loss)/dw = [2, 2] per thread.
    // After N threads: w.grad() == [N*2, N*2].
    constexpr int N = 4;
    Tensor w = Tensor::from_data<double>({3.0, 1.0}, {2}, be, /*requires_grad=*/true);

    // Spin barrier: all threads build their graphs first, then all call backward()
    // simultaneously to maximise contention on w's GradAccumulator.
    std::atomic<int> built{0};
    std::atomic<bool> go{false};

    std::vector<std::thread> threads;
    threads.reserve(N);

    for (int i = 0; i < N; ++i) {
        threads.emplace_back([&]() {
            // Each thread builds its OWN computation graph. w is the shared leaf;
            // scale and loss are thread-private (each call to from_data/mul/sum
            // creates fresh Operation objects owned by this thread).
            Tensor scale = Tensor::from_data<double>({2.0, 2.0}, {2}, be);
            Tensor loss  = w.mul(scale).sum();

            built.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) { /* spin */ }

            loss.backward();
        });
    }

    // Wait until all threads have built their graphs before releasing them.
    while (built.load(std::memory_order_acquire) < N) { /* spin */ }
    go.store(true, std::memory_order_release);

    for (auto& t : threads) t.join();

    // Each thread contributed dw = [2, 2]; total = [N*2, N*2].
    const double expected = static_cast<double>(N) * 2.0;
    CHECK_NEAR(w.grad().at({0}), expected, 1e-9);
    CHECK_NEAR(w.grad().at({1}), expected, 1e-9);
}

// ── Test 3: backward() and zero_grad() interleaved ───────────────────────────

void test_zero_grad_interleaved_with_backward() {
    std::cout << "[ThreadSafety 3] backward() + zero_grad() interleaved — no UB, valid shape\n";
    Backend& be = cpu_backend();

    // Run many trials to stress different scheduling outcomes.
    for (int trial = 0; trial < 32; ++trial) {
        Tensor w    = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be,
                                                /*requires_grad=*/true);
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

        // Result is non-deterministic: zero_grad() may win or backward() may win.
        // All we require is structural validity — no crash, no corrupt shape.
        Tensor g = w.grad();
        if (g.defined()) {
            CHECK(g.shape() == w.shape());
        }
        // If g is undefined, zero_grad() ran last — valid outcome.
    }

    // Count the overall test as a single pass/fail unit.
    ++tests_run;
    ++tests_passed;
}

// ── Run all ───────────────────────────────────────────────────────────────────

void run_thread_safety_tests() {
    test_grad_safe_concurrent_reads();
    test_concurrent_backward_separate_graphs();
    test_zero_grad_interleaved_with_backward();
}

} // namespace otter::test
