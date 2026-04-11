#pragma once

#include "otter/kernel/backend.h"

namespace otter {

// Returns the pre-built CUDA backend.
// Initialised once on first call (C++11 magic static — thread-safe).
// The returned reference is valid for the lifetime of the program.
//
// Requires OTTER_CUDA=ON at build time. Calling this without a CUDA-capable
// device results in a runtime error from the CUDAMemoryManager constructor.
//
//   Tensor a = Tensor::zeros({2, 3}, otter::cuda_backend());
//
// Implemented in src/backends/cuda_backend.cpp.
[[nodiscard]] Backend& cuda_backend();

} // namespace otter
