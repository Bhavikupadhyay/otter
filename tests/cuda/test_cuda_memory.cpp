#include "../utils/test_utils.h"

#include <cstddef>

#include "otter/backends/cuda.h"
#include "otter/tensor.h"

namespace otter::test {

void test_cuda_A1_bytes_allocated_starts_at_zero() {
    std::cout << "[CUDA A1] bytes_allocated is 0 when no tensors are live\n";
    CHECK(cuda_backend().memory_manager()->bytes_allocated() == 0);
}

void test_cuda_A2_allocate_increments_free_decrements() {
    std::cout << "[CUDA A2] allocate increments bytes_allocated; free returns it to 0\n";
    const std::size_t before = cuda_backend().memory_manager()->bytes_allocated();
    {
        Tensor t = Tensor::zeros({16}, cuda_backend());
        CHECK(cuda_backend().memory_manager()->bytes_allocated() > before);
    }
    CHECK(cuda_backend().memory_manager()->bytes_allocated() == before);
}

void test_cuda_A3_multiple_tensors_accumulate_and_release() {
    std::cout << "[CUDA A3] multiple tensors accumulate bytes; all released on destruct\n";
    const std::size_t start = cuda_backend().memory_manager()->bytes_allocated();
    {
        Tensor a = Tensor::zeros({4},  cuda_backend());
        Tensor b = Tensor::zeros({8},  cuda_backend());
        Tensor c = Tensor::zeros({16}, cuda_backend());
        // At minimum 4+8+16 = 28 doubles = 224 bytes extra
        CHECK(cuda_backend().memory_manager()->bytes_allocated()
              >= start + 28 * sizeof(double));
    }
    CHECK(cuda_backend().memory_manager()->bytes_allocated() == start);
}

void test_cuda_A4_release_cache_does_not_crash() {
    std::cout << "[CUDA A4] release_cache() flushes pool and does not crash\n";
    cuda_backend().memory_manager()->release_cache();
    CHECK(cuda_backend().memory_manager()->bytes_allocated() == 0);
}

void test_cuda_A5_small_alloc_reserved_equals_allocated() {
    std::cout << "[CUDA A5] small alloc (< 1 MB): bytes_reserved == bytes_allocated — not pooled\n";
    // Release pool before measuring so prior tests' cached segments don't interfere.
    cuda_backend().memory_manager()->release_cache();
    Tensor t = Tensor::zeros({32}, cuda_backend());
    // {32} doubles = 256 bytes — small path, freed immediately, no pool overhead.
    CHECK(cuda_backend().memory_manager()->bytes_reserved() ==
          cuda_backend().memory_manager()->bytes_allocated());
}

void test_cuda_A6_large_alloc_pool_hit_does_not_grow_reserved() {
    std::cout << "[CUDA A6] large alloc pool hit: bytes_reserved stable on re-allocation\n";
    // Use a large tensor (>= 1 MB) so it enters the pool.
    // 1 MB / sizeof(double) = 131072 elements.
    constexpr std::size_t n = 131072;
    cuda_backend().memory_manager()->release_cache();

    {
        Tensor t = Tensor::zeros({n}, cuda_backend());
        // First allocation: reserved grows.
        CHECK(cuda_backend().memory_manager()->bytes_reserved() > 0);
    }
    // Tensor destroyed: bytes_allocated_ == 0, segment cached in pool.
    CHECK(cuda_backend().memory_manager()->bytes_allocated() == 0);
    const std::size_t reserved_after_free = cuda_backend().memory_manager()->bytes_reserved();
    CHECK(reserved_after_free > 0); // pool still holds the segment

    {
        Tensor t2 = Tensor::zeros({n}, cuda_backend());
        // Pool hit: reserved must not grow — we reuse the cached segment.
        CHECK(cuda_backend().memory_manager()->bytes_reserved() == reserved_after_free);
        CHECK(cuda_backend().memory_manager()->bytes_allocated() > 0);
    }

    // Clean up.
    cuda_backend().memory_manager()->release_cache();
}

void test_cuda_A7_release_cache_flushes_pool_reserved_to_zero() {
    std::cout << "[CUDA A7] release_cache() after large alloc+free: bytes_reserved == 0\n";
    constexpr std::size_t n = 131072; // >= 1 MB — large path
    {
        Tensor t = Tensor::zeros({n}, cuda_backend());
        (void)t;
    }
    // Segment is in the pool: bytes_allocated == 0 but bytes_reserved > 0.
    CHECK(cuda_backend().memory_manager()->bytes_allocated() == 0);
    CHECK(cuda_backend().memory_manager()->bytes_reserved() > 0);

    cuda_backend().memory_manager()->release_cache();

    // After flush: both counters reach 0.
    CHECK(cuda_backend().memory_manager()->bytes_allocated() == 0);
    CHECK(cuda_backend().memory_manager()->bytes_reserved() == 0);
}

void run_cuda_memory_tests() {
    test_cuda_A1_bytes_allocated_starts_at_zero();
    test_cuda_A2_allocate_increments_free_decrements();
    test_cuda_A3_multiple_tensors_accumulate_and_release();
    test_cuda_A4_release_cache_does_not_crash();
    test_cuda_A5_small_alloc_reserved_equals_allocated();
    test_cuda_A6_large_alloc_pool_hit_does_not_grow_reserved();
    test_cuda_A7_release_cache_flushes_pool_reserved_to_zero();
}

} // namespace otter::test
