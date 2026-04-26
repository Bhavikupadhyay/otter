#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace otter {

// Launch parameters for CUDA kernels.
// Defaults reproduce the original hardcoded behaviour (block=256, no stream,
// sync after every launch). Override per-engine or per-call as needed.
struct LaunchSpec {
    std::size_t  block_size = 256;
    cudaStream_t stream     = nullptr;
    bool         sync_after = true;
};

} // namespace otter
