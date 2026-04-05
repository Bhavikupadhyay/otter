#pragma once

#include <cstddef>
#include <vector>

namespace otter::detail {

// Returns the contiguous (row-major) strides for a given shape.
// stride[i] = product of shape[i+1 .. ndim-1].
// Example: shape {3, 4, 5} → strides {20, 5, 1}.
inline std::vector<std::size_t> contiguous_strides(
    const std::vector<std::size_t>& shape)
{
    std::vector<std::size_t> s(shape.size());
    std::size_t v = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        s[static_cast<std::size_t>(i)] = v;
        v *= shape[static_cast<std::size_t>(i)];
    }
    return s;
}

} // namespace otter::detail
