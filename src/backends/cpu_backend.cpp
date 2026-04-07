#include "otter/backends/cpu.h"
#include "memory/cpu_memory_manager.h"
#include "kernels/cpu_kernel_engine.h"

namespace otter {

// cpu_backend() — returns the singleton CPU Backend.
//
// C++11 magic static: initialised exactly once, thread-safe.
// The Backend owns CPUMemoryManager and CPUKernelEngine for their lifetimes.
// All Tensors allocated via cpu_backend() keep a non-owning Backend* —
// the singleton lives for the duration of the process, so no dangling pointer.
Backend& cpu_backend() {
    static Backend instance(
        std::make_unique<CPUMemoryManager>(),
        std::make_unique<CPUKernelEngine>()
    );
    return instance;
}

} // namespace otter
