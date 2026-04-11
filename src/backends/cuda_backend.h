#pragma once

// Internal header — not part of the public include/otter/ surface.

#include "otter/kernel/backend.h"
#include "cuda_stream.h"

namespace otter {

// CUDABackend — Backend subclass for CUDA devices.
//
// Constructs with CUDAMemoryManager + CUDAKernelEngine.
// Owns a CUDAStream; default_stream() returns a non-owning pointer to it.
// Lifetime: singleton (via cuda_backend()). Outlives all CUDA Tensors.
class CUDABackend final : public Backend {
public:
    CUDABackend();

    [[nodiscard]] Stream* default_stream() noexcept override { return &stream_; }

private:
    CUDAStream stream_;  // owned; constructed after mm/ke are moved into Backend
};

} // namespace otter
