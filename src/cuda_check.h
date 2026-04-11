#pragma once

// Private header — include from any CUDA translation unit (src/*.cpp / src/**/*.cu).
// Provides OTTER_CUDA_CHECK: throws std::runtime_error on CUDA API failure.
// For noexcept callers use the raw CUDA API and check via assert instead.

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#define OTTER_CUDA_CHECK(call)                                              \
    do {                                                                    \
        cudaError_t _err = (call);                                          \
        if (_err != cudaSuccess)                                            \
            throw std::runtime_error(                                       \
                std::string("CUDA error: ") + ::cudaGetErrorString(_err));  \
    } while (0)
