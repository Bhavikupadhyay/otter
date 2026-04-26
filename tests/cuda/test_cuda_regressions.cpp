#include "../utils/test_utils.h"

#include <cstddef>
#include <stdexcept>

#include "otter/backends/cuda.h"
#include "otter/tensor.h"

// Private headers — accessible because tests/CMakeLists.txt adds PROJECT_SOURCE_DIR/src.
#include "backends/cuda_stream.h"
#include "kernels/cuda_kernel_engine.h"
#include "memory/cuda_memory_manager.h"

namespace otter::test {

void test_cuda_R1_engine_default_stream_matches_backend_stream() {
    std::cout << "[CUDA R1] kernel engine launch stream matches backend default stream\n";
    auto* engine = dynamic_cast<CUDAKernelEngine*>(cuda_backend().kernel_engine());
    auto* stream = static_cast<CUDAStream*>(cuda_backend().default_stream());
    CHECK(engine != nullptr);
    CHECK(stream != nullptr);
    CHECK(engine->default_spec_.stream == stream->raw());
}

void test_cuda_R2_sync_after_false_strided_path_still_computes_correctly() {
    std::cout << "[CUDA R2] sync_after=false: strided CUDA path still produces correct values\n";
    auto* engine = dynamic_cast<CUDAKernelEngine*>(cuda_backend().kernel_engine());
    CHECK(engine != nullptr);

    const LaunchSpec saved = engine->default_spec_;
    engine->default_spec_.sync_after = false;

    Tensor a = Tensor::from_data<double>({1, 2, 3, 4, 5, 6}, {2, 3}, cuda_backend());
    Tensor t = a.transpose(0, 1);  // non-contiguous, forces strided unary path
    Tensor n = t.neg();

    // release_cache() is a full device fence for the current CUDA allocator.
    cuda_backend().memory_manager()->release_cache();

    CHECK_NEAR(n.at({0, 0}), -1.0, 1e-12);
    CHECK_NEAR(n.at({0, 1}), -4.0, 1e-12);
    CHECK_NEAR(n.at({1, 0}), -2.0, 1e-12);
    CHECK_NEAR(n.at({1, 1}), -5.0, 1e-12);
    CHECK_NEAR(n.at({2, 0}), -3.0, 1e-12);
    CHECK_NEAR(n.at({2, 1}), -6.0, 1e-12);

    engine->default_spec_ = saved;
}

void test_cuda_R3_release_rejects_unsupported_alignment() {
    std::cout << "[CUDA R3] allocator rejects alignments above 256 bytes\n";
#ifdef NDEBUG
    CUDAMemoryManager mm;
    bool threw = false;
    try {
        std::byte* ptr = mm.allocate(8, 512);
        mm.free(ptr);
    } catch (const std::invalid_argument&) {
        threw = true;
    } catch (...) {
        threw = false;
    }
    CHECK(threw);
#else
    CHECK(true);
#endif
}

void run_cuda_regression_tests() {
    test_cuda_R1_engine_default_stream_matches_backend_stream();
    test_cuda_R2_sync_after_false_strided_path_still_computes_correctly();
    test_cuda_R3_release_rejects_unsupported_alignment();
}

} // namespace otter::test
