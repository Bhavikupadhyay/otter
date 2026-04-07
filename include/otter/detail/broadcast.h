#pragma once

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace otter::detail {

// Compute the output shape of a NumPy-style broadcast between two shapes.
// Aligns from the right; each dimension must be equal, or one of them must be 1.
// Throws std::runtime_error if the shapes are not broadcastable.
inline std::vector<std::size_t> broadcast_shape(
    const std::vector<std::size_t>& a,
    const std::vector<std::size_t>& b)
{
    const std::size_t ndim = std::max(a.size(), b.size());
    std::vector<std::size_t> out(ndim);
    for (std::size_t i = 0; i < ndim; ++i) {
        const std::size_t ai = (i < ndim - a.size()) ? 1 : a[i - (ndim - a.size())];
        const std::size_t bi = (i < ndim - b.size()) ? 1 : b[i - (ndim - b.size())];
        if (ai != bi && ai != 1 && bi != 1)
            throw std::runtime_error(
                "broadcast_shape: shapes are not broadcastable at dimension " +
                std::to_string(i) + " (sizes " + std::to_string(ai) +
                " and " + std::to_string(bi) + ")");
        out[i] = std::max(ai, bi);
    }
    return out;
}

// Compute the strides needed to broadcast a tensor of orig_shape/orig_stride
// to target_shape without copying data.
//
// Dimensions being broadcast (size-1 or prepended) receive stride 0.
// The kernel reads the same element repeatedly for those dimensions.
//
// Precondition: orig_shape and target_shape must be broadcast-compatible
// (i.e. broadcast_shape(orig_shape, target_shape) == target_shape).
// Call broadcast_shape() first to validate; it throws on incompatible shapes.
inline std::vector<std::size_t> broadcast_strides(
    const std::vector<std::size_t>& orig_shape,
    const std::vector<std::size_t>& orig_stride,
    const std::vector<std::size_t>& target_shape)
{
    const std::size_t tndim  = target_shape.size();
    assert(orig_shape.size() <= tndim &&
           "broadcast_strides: orig_shape cannot be larger than target_shape");
    // Validate broadcast compatibility: every non-prepended original dim must
    // equal the target dim or be 1. Caller must have called broadcast_shape() first.
    for (std::size_t i = 0; i < orig_shape.size(); ++i) {
        assert((orig_shape[i] == target_shape[i + (tndim - orig_shape.size())] ||
                orig_shape[i] == 1) &&
               "broadcast_strides: shapes are not broadcast-compatible at this dimension");
    }
    const std::size_t offset = tndim - orig_shape.size();
    std::vector<std::size_t> strides(tndim, 0);
    for (std::size_t i = 0; i < tndim; ++i) {
        if (i < offset) {
            strides[i] = 0;
        } else {
            const std::size_t orig_dim = i - offset;
            strides[i] = (orig_shape[orig_dim] == 1) ? 0 : orig_stride[orig_dim];
        }
    }
    return strides;
}

} // namespace otter::detail
