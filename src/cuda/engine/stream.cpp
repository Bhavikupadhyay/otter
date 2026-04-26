#include "cuda/engine/stream.h"

#include <cassert>

#include "cuda/cuda_check.h"

namespace otter {

CUDAStream::CUDAStream() {
    OTTER_CUDA_CHECK(::cudaStreamCreate(&stream_));
}

CUDAStream::~CUDAStream() noexcept {
    cudaError_t err = ::cudaStreamDestroy(stream_);
    assert(err == cudaSuccess && "CUDAStream::~CUDAStream: cudaStreamDestroy failed");
    (void)err;
}

} // namespace otter
