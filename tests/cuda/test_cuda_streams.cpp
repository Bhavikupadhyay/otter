#include "../utils/test_utils.h"

#include "otter/backends/cuda.h"
#include "otter/core/stream.h"

// Private header — accessible because tests/CMakeLists.txt adds PROJECT_SOURCE_DIR/src.
#include "backends/cuda_stream.h"

namespace otter::test {

void test_cuda_H1_default_stream_not_null() {
    std::cout << "[CUDA H1] default_stream() returns non-null\n";
    CHECK(cuda_backend().default_stream() != nullptr);
}

void test_cuda_H2_stream_raw_handle_valid() {
    std::cout << "[CUDA H2] CUDAStream::raw() returns a valid (non-null) cudaStream_t\n";
    Stream* s  = cuda_backend().default_stream();
    auto*   cs = static_cast<CUDAStream*>(s);
    CHECK(cs != nullptr);
    CHECK(cs->raw() != nullptr);
}

void run_cuda_stream_tests() {
    test_cuda_H1_default_stream_not_null();
    test_cuda_H2_stream_raw_handle_valid();
}

} // namespace otter::test
