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
    std::cout << "[CUDA A4] release_cache() (== cudaDeviceSynchronize) does not crash\n";
    cuda_backend().memory_manager()->release_cache();
    CHECK(cuda_backend().memory_manager()->bytes_allocated() == 0);
}

void test_cuda_A5_bytes_reserved_equals_allocated() {
    std::cout << "[CUDA A5] bytes_reserved() == bytes_allocated() — no pool overhead\n";
    Tensor t = Tensor::zeros({32}, cuda_backend());
    CHECK(cuda_backend().memory_manager()->bytes_reserved() ==
          cuda_backend().memory_manager()->bytes_allocated());
}

void run_cuda_memory_tests() {
    test_cuda_A1_bytes_allocated_starts_at_zero();
    test_cuda_A2_allocate_increments_free_decrements();
    test_cuda_A3_multiple_tensors_accumulate_and_release();
    test_cuda_A4_release_cache_does_not_crash();
    test_cuda_A5_bytes_reserved_equals_allocated();
}

} // namespace otter::test
