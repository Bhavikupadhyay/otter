#pragma once

#include <cstddef>

namespace otter {

enum class DType { Float64 };

namespace dtype_utils {

constexpr std::size_t size_of(DType d) noexcept {
    switch (d) {
        case DType::Float64: return sizeof(double);
    }
    return 0;
}

inline const char* name(DType d) noexcept {
    switch (d) {
        case DType::Float64: return "float64";
    }
    return "unknown";
}

template<typename T> struct dtype_of;
template<> struct dtype_of<double> {
    static constexpr DType value = DType::Float64;
};

} // namespace dtype_utils
} // namespace otter
