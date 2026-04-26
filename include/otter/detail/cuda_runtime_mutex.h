#pragma once

#include <mutex>

namespace otter::detail {

// Serializes CUDA runtime API calls across host threads.
//
// Guards cudaMalloc/cudaFree (for persistent tensor buffers and per-launch
// metadata), cudaMemcpy, cudaMemset, cudaStreamSynchronize, and
// cudaDeviceSynchronize calls. These are not individually thread-safe when
// called concurrently from multiple host threads against the same CUDA context.
//
// Scope: narrowed from the original cudaMallocManaged era. Host-read paths
// (cuda_element_read, cuda_bulk_host_read) still use cudaDeviceSynchronize
// because they must observe all streams. Compute kernel syncs use
// cudaStreamSynchronize, scoped to the engine's stream only.
//
// Future: can be removed from per-launch metadata paths once those switch to
// cudaMallocAsync / cudaFreeAsync (stream-ordered, inherently thread-safe).
inline std::mutex& cuda_runtime_mutex() {
    static std::mutex mutex;
    return mutex;
}

} // namespace otter::detail
