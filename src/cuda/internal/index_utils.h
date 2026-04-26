#pragma once

// Device-side index helpers — included by any .cu file that needs
// flat-index → memory-offset conversion for strided tensors.
//
// All functions fuse decomposition and offset accumulation in a single pass
// so no intermediate coords[] array is needed. This removes any implicit
// upper bound on tensor rank.
//
// No max-ndim limit is enforced here. If a rank ceiling is ever introduced
// (e.g. for cuDNN interop or fixed device descriptor structs), it belongs
// at Tensor construction time, not inside backend kernels.

#include <cstddef>

// Compute the flat memory offset for a single tensor given a logical flat index.
// Handles stride-0 broadcast dims: a zero stride contributes 0 regardless of coord.
__device__ __forceinline__
std::size_t flat_to_offset(std::size_t flat,
                            const std::size_t* __restrict__ shape,
                            const std::size_t* __restrict__ strides,
                            std::size_t ndim) noexcept {
    std::size_t off = 0;
    for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
        const std::size_t ud    = static_cast<std::size_t>(d);
        const std::size_t coord = flat % shape[ud];
        flat                   /= shape[ud];
        off                    += coord * strides[ud];
    }
    return off;
}

// Compute offsets into two tensors (a and b) that share the same logical shape.
// One decomposition pass, two offset accumulators — used by binary strided kernels.
// Handles stride-0 broadcast dims in either tensor independently.
__device__ __forceinline__
void flat_to_two_offsets(std::size_t flat,
                          const std::size_t* __restrict__ shape,
                          const std::size_t* __restrict__ strides_a,
                          const std::size_t* __restrict__ strides_b,
                          std::size_t ndim,
                          std::size_t& off_a,
                          std::size_t& off_b) noexcept {
    off_a = 0;
    off_b = 0;
    for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
        const std::size_t ud    = static_cast<std::size_t>(d);
        const std::size_t coord = flat % shape[ud];
        flat                   /= shape[ud];
        off_a                  += coord * strides_a[ud];
        off_b                  += coord * strides_b[ud];
    }
}
