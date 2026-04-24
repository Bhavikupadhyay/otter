#pragma once

#include <mutex>

namespace otter::detail {

// Serializes CUDA runtime API calls across host threads.
//
// The CUDA runtime in this codebase performs synchronous work in several
// places (cudaMallocManaged, cudaMalloc/cudaFree, cudaDeviceSynchronize).
// Those calls are safe to coordinate through a single process-wide mutex so
// thread-heavy tests do not race inside the driver/runtime.
inline std::mutex& cuda_runtime_mutex() {
    static std::mutex mutex;
    return mutex;
}

} // namespace otter::detail
