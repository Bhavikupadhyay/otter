#pragma once

#include <cuda_runtime.h>

#include "otter/core/stream.h"

namespace otter {

// CUDAStream — wraps a single cudaStream_t.
//
// Created and owned by CUDABackend. Lifetime tied to the backend singleton.
// default_stream() returns a non-owning pointer to the CUDABackend's instance.
class CUDAStream final : public Stream {
public:
    CUDAStream();
    ~CUDAStream() noexcept override;

    CUDAStream(const CUDAStream&)            = delete;
    CUDAStream& operator=(const CUDAStream&) = delete;
    CUDAStream(CUDAStream&&)                 = delete;
    CUDAStream& operator=(CUDAStream&&)      = delete;

    [[nodiscard]] cudaStream_t raw() const noexcept { return stream_; }

private:
    cudaStream_t stream_{};
};

} // namespace otter
