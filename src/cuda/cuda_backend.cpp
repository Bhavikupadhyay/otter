#include "cuda/cuda_backend.h"
#include "otter/backends/cuda.h"
#include "cuda/memory/cuda_memory_manager.h"
#include "cuda/engine/cuda_kernel_engine.h"

#include <cassert>
#include <memory>

namespace otter {

CUDABackend::CUDABackend()
    : Backend(std::make_unique<CUDAMemoryManager>(),
              std::make_unique<CUDAKernelEngine>())
{
    auto* engine = dynamic_cast<CUDAKernelEngine*>(kernel_engine());
    assert(engine != nullptr && "CUDABackend: kernel_engine() must be CUDAKernelEngine");
    engine->default_spec_.stream = stream_.raw();
}

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
