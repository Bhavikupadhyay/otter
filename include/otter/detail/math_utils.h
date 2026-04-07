#pragma once

#include <cstddef>
#include <cstdint>

namespace otter::detail {

inline std::size_t next_power_of_2(std::size_t n) noexcept {
    if (n == 0) return 1;
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    if constexpr (sizeof(std::size_t) >= 8) n |= n >> 32;
    return ++n;
}

inline constexpr bool is_power_of_2(std::size_t n) noexcept {
    return n > 0 && (n & (n - 1)) == 0;
}

inline void* align_ptr(void* ptr, std::size_t alignment) noexcept {
    const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(ptr);
    return reinterpret_cast<void*>((addr + alignment - 1) & ~(alignment - 1));
}

} // namespace otter::detail
