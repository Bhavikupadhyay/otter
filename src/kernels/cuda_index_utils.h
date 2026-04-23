#pragma once

// Device-side coordinate helpers — included by any .cu file that needs
// flat-index ↔ multi-dimensional coordinate conversion.
//
// Max supported ndim: 8 (matches the on-stack coords[8] arrays throughout).
// All functions are __device__ __forceinline__ — zero call overhead.

#include <cstddef>

// Decompose a flat linear index into per-dimension coordinates.
// Fills coords[0..ndim-1] in-place. coords must be at least ndim elements.
__device__ __forceinline__
void flat_to_coords(std::size_t flat,
                    const std::size_t* __restrict__ shape,
                    std::size_t ndim,
                    std::size_t* __restrict__ coords) noexcept {
    for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
        const std::size_t ud = static_cast<std::size_t>(d);
        coords[ud] = flat % shape[ud];
        flat      /= shape[ud];
    }
}

// Compute byte (element) offset from coords and strides.
// For stride-0 broadcast dims the contribution is 0 — correct by construction.
__device__ __forceinline__
std::size_t coords_to_offset(const std::size_t* __restrict__ coords,
                              const std::size_t* __restrict__ strides,
                              std::size_t ndim) noexcept {
    std::size_t off = 0;
    for (std::size_t d = 0; d < ndim; ++d) off += coords[d] * strides[d];
    return off;
}
