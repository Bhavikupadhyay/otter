#include "cuda_backend.h"
#include "otter/backends/cuda.h"
#include "memory/cuda_memory_manager.h"
#include "kernels/cuda_kernel_engine.h"

#include <memory>

namespace otter {

CUDABackend::CUDABackend()
    : Backend(std::make_unique<CUDAMemoryManager>(),
              std::make_unique<CUDAKernelEngine>())
{}

// cuda_backend() — returns the singleton CUDA Backend.
//
// C++11 magic static: initialised exactly once, thread-safe.
// The Backend owns CUDAMemoryManager and CUDAKernelEngine for their lifetimes.
// All Tensors allocated via cuda_backend() keep a non-owning Backend* —
// the singleton lives for the duration of the process, so no dangling pointer.
Backend& cuda_backend() {
    static CUDABackend instance;
    return instance;
}

} // namespace otter
