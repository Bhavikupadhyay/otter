#include "test_utils.h"

#include "otter/backends/cpu.h"
#include "otter/backends/cuda.h"
#include "otter/tensor.h"

namespace otter::test {

void run_cuda_tests() {
    // ── Test 1: cuda_backend() returns a valid backend on Device::CUDA ────────
    {
        std::cout << "[CUDA Test 1] cuda_backend() device is Device::CUDA\n";
        Backend& b = cuda_backend();
        CHECK(b.device() == Device::CUDA);
    }

    // ── Test 2: Tensor::zeros allocates without throwing ──────────────────────
    {
        std::cout << "[CUDA Test 2] Tensor::zeros on CUDA backend allocates\n";
        Tensor t = Tensor::zeros({4}, cuda_backend());
        CHECK(t.defined());
        CHECK(t.numel() == 4);
    }

    // ── Test 3: bytes_allocated increments and returns to 0 after destruct ────
    {
        std::cout << "[CUDA Test 3] bytes_allocated() tracks live allocation\n";
        const std::size_t before = cuda_backend().memory_manager()->bytes_allocated();
        {
            Tensor t = Tensor::zeros({4}, cuda_backend());
            CHECK(cuda_backend().memory_manager()->bytes_allocated() > before);
        }
        CHECK(cuda_backend().memory_manager()->bytes_allocated() == before);
    }

    // ── Test 4: from_data reads back correctly via at() ───────────────────────
    {
        std::cout << "[CUDA Test 4] from_data reads back correctly via at()\n";
        Tensor t = Tensor::from_data<double>({1.0, 2.0, 3.0, 4.0}, {4}, cuda_backend());
        CHECK_NEAR(t.at({0}), 1.0, 1e-10);
        CHECK_NEAR(t.at({1}), 2.0, 1e-10);
        CHECK_NEAR(t.at({2}), 3.0, 1e-10);
        CHECK_NEAR(t.at({3}), 4.0, 1e-10);
    }

    // ── Test 5: fill_() writes and reads back correctly ───────────────────────
    {
        std::cout << "[CUDA Test 5] fill_() writes via kernel; at() reads back\n";
        Tensor t = Tensor::zeros({8}, cuda_backend());
        t.fill_(3.14);
        for (std::size_t i = 0; i < 8; ++i)
            CHECK_NEAR(t.at({i}), 3.14, 1e-10);
    }

    // ── Test 6: default_stream() returns non-null ──────────────────────────────
    {
        std::cout << "[CUDA Test 6] default_stream() is non-null\n";
        CHECK(cuda_backend().default_stream() != nullptr);
    }

    // ── Test 7: to(Device::CPU) reads back correct values ─────────────────────
    {
        std::cout << "[CUDA Test 7] to(Device::CPU) transfers values to host\n";
        Tensor t = Tensor::from_data<double>({5.0, 6.0, 7.0, 8.0}, {4}, cuda_backend());
        Tensor h = t.to(Device::CPU);
        CHECK(h.backend().device() == Device::CPU);
        CHECK_NEAR(h.at({0}), 5.0, 1e-10);
        CHECK_NEAR(h.at({1}), 6.0, 1e-10);
        CHECK_NEAR(h.at({2}), 7.0, 1e-10);
        CHECK_NEAR(h.at({3}), 8.0, 1e-10);
    }

    // ── Test 8: cpu() convenience method ──────────────────────────────────────
    {
        std::cout << "[CUDA Test 8] cpu() convenience method\n";
        Tensor t = Tensor::zeros({3}, cuda_backend());
        t.fill_(9.0);
        Tensor h = t.cpu();
        CHECK(h.backend().device() == Device::CPU);
        CHECK_NEAR(h.at({0}), 9.0, 1e-10);
        CHECK_NEAR(h.at({2}), 9.0, 1e-10);
    }

    // ── Test 9: cuda() on a CPU tensor produces a CUDA tensor ─────────────────
    {
        std::cout << "[CUDA Test 9] cuda() moves a CPU tensor to CUDA\n";
        Tensor h = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, cpu_backend());
        Tensor d = h.cuda();
        CHECK(d.backend().device() == Device::CUDA);
        CHECK_NEAR(d.at({0}), 1.0, 1e-10);
        CHECK_NEAR(d.at({1}), 2.0, 1e-10);
        CHECK_NEAR(d.at({2}), 3.0, 1e-10);
    }

    // ── Test 10: to() on same device is a no-copy fast path ───────────────────
    {
        std::cout << "[CUDA Test 10] to(Device::CUDA) on CUDA tensor returns same object\n";
        Tensor t = Tensor::zeros({4}, cuda_backend());
        t.fill_(42.0);
        Tensor same = t.to(Device::CUDA);
        CHECK(same.backend().device() == Device::CUDA);
        // Verify values are intact (fast path — no allocation)
        CHECK_NEAR(same.at({0}), 42.0, 1e-10);
        CHECK_NEAR(same.at({3}), 42.0, 1e-10);
    }

    // ── Test 11: round-trip CPU → CUDA → CPU preserves values ────────────────
    {
        std::cout << "[CUDA Test 11] round-trip CPU → CUDA → CPU preserves values\n";
        Tensor cpu0 = Tensor::from_data<double>({1.1, 2.2, 3.3, 4.4}, {4}, cpu_backend());
        Tensor gpu  = cpu0.cuda();
        Tensor cpu1 = gpu.cpu();
        CHECK(cpu1.backend().device() == Device::CPU);
        CHECK_NEAR(cpu1.at({0}), 1.1, 1e-10);
        CHECK_NEAR(cpu1.at({1}), 2.2, 1e-10);
        CHECK_NEAR(cpu1.at({2}), 3.3, 1e-10);
        CHECK_NEAR(cpu1.at({3}), 4.4, 1e-10);
    }
}

} // namespace otter::test
