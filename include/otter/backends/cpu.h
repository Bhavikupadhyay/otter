#pragma once

#include "otter/kernel/backend.h"

namespace otter {

// Returns the pre-built CPU backend.
// Initialised once on first call (C++11 magic static — thread-safe).
// The returned reference is valid for the lifetime of the program.
//
// This is the only entry point users need for CPU computation:
//   Tensor a = Tensor::zeros({2, 3}, otter::cpu_backend());
//
// Implemented in src/backends/cpu_backend.cpp.
[[nodiscard]] Backend& cpu_backend();

} // namespace otter
