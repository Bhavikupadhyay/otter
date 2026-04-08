#pragma once

// otter/debug.h — opt-in diagnostic utilities.
//
// These are O(n) host-side scans intended for debugging and test validation.
// None of these are on the training hot path. Do not call inside kernels or
// backward passes.
//
// All functions require a defined() tensor. They throw std::runtime_error
// on undefined input so callers get a clear message rather than a crash.

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "otter/tensor.h"

namespace otter {

// Returns the shape as a human-readable string, e.g. "[2, 3]".
inline std::string shape_str(const Tensor& t) {
    std::ostringstream oss;
    oss << "[";
    const auto& s = t.shape();
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (i) oss << ", ";
        oss << s[i];
    }
    oss << "]";
    return oss.str();
}

// Returns the dtype as a human-readable string, e.g. "Float64".
inline std::string dtype_str(const Tensor& t) {
    switch (t.dtype()) {
        case DType::Float64: return "Float64";
        default:             return "unknown";
    }
}

// Returns true if any element of t is NaN.
// Uses IEEE 754 property: NaN != NaN.
inline bool has_nan(const Tensor& t) {
    if (!t.defined())
        throw std::runtime_error("has_nan: tensor is undefined");
    const auto v = t.to_vector<double>();
    for (double x : v)
        if (std::isnan(x)) return true;
    return false;
}

// Returns true if any element of t is ±inf.
inline bool has_inf(const Tensor& t) {
    if (!t.defined())
        throw std::runtime_error("has_inf: tensor is undefined");
    const auto v = t.to_vector<double>();
    for (double x : v)
        if (std::isinf(x)) return true;
    return false;
}

// Returns max |a[i] - b[i]| over all elements.
// a and b must have the same shape.
inline double max_abs_diff(const Tensor& a, const Tensor& b) {
    if (!a.defined() || !b.defined())
        throw std::runtime_error("max_abs_diff: input tensor is undefined");
    if (a.shape() != b.shape()) {
        std::ostringstream msg;
        msg << "max_abs_diff: shape mismatch ("
            << shape_str(a) << " vs " << shape_str(b) << ")";
        throw std::runtime_error(msg.str());
    }
    const auto va = a.to_vector<double>();
    const auto vb = b.to_vector<double>();
    double maxd = 0.0;
    for (std::size_t i = 0; i < va.size(); ++i) {
        double d = std::abs(va[i] - vb[i]);
        if (d > maxd) maxd = d;
    }
    return maxd;
}

} // namespace otter
