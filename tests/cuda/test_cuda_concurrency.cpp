#include "../utils/test_utils.h"

#include <atomic>
#include <thread>
#include <vector>

#include "otter/backends/cuda.h"
#include "otter/tensor.h"

namespace otter::test {

void test_cuda_O1_concurrent_grad_reads_after_backward() {
    std::cout << "[CUDA O1] N threads read w.grad() concurrently after backward — no data race\n";
    // backward() completes on the main thread first; then all reader threads start.
    Tensor w = Tensor::from_data<double>({5.0, 6.0}, {2}, cuda_backend(), /*requires_grad=*/true);
    w.add(Tensor::zeros({2}, cuda_backend())).sum().backward();

    constexpr std::size_t N = 8;
    std::vector<double> val0(N, -1.0);
    std::vector<double> val1(N, -1.0);
    std::vector<std::thread> threads;
    threads.reserve(N);

    for (std::size_t i = 0; i < N; ++i) {
        threads.emplace_back([&, i]() {
            val0[i] = w.grad().at({0});
            val1[i] = w.grad().at({1});
        });
    }
    for (auto& t : threads) t.join();

    for (std::size_t i = 0; i < N; ++i) {
        CHECK_NEAR(val0[i], 1.0, 1e-9);
        CHECK_NEAR(val1[i], 1.0, 1e-9);
    }
}

void test_cuda_O2_concurrent_backward_shared_cuda_leaf() {
    std::cout << "[CUDA O2] N threads backward over shared CUDA leaf — grad accumulates correctly\n";
    // Each thread: loss = sum(w * [2,2]) → d(loss)/dw = {2,2} per thread.
    // After N threads: w.grad() == {N*2, N*2}.
    constexpr int N = 4;
    Tensor w = Tensor::from_data<double>({3.0, 1.0}, {2}, cuda_backend(), /*requires_grad=*/true);

    std::atomic<int>  built{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;
    threads.reserve(N);

    for (int i = 0; i < N; ++i) {
        threads.emplace_back([&]() {
            Tensor scale = Tensor::from_data<double>({2.0, 2.0}, {2}, cuda_backend());
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

void run_cuda_concurrency_tests() {
    test_cuda_O1_concurrent_grad_reads_after_backward();
    test_cuda_O2_concurrent_backward_shared_cuda_leaf();
}

} // namespace otter::test
