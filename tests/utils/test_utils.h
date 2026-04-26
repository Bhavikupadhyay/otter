#pragma once

#include <cmath>
#include <iostream>
#include <string>

#include "otter/tensor.h"

namespace otter::test {

inline int tests_run    = 0;
inline int tests_passed = 0;

} // namespace otter::test

#define CHECK(cond)                                                        \
    do {                                                                   \
        ++otter::test::tests_run;                                          \
        if (!(cond)) {                                                     \
            std::cerr << "  FAIL  " #cond                                  \
                      << "  (" __FILE__ ":" << __LINE__ << ")\n";         \
        } else {                                                           \
            ++otter::test::tests_passed;                                   \
        }                                                                  \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                              \
    CHECK(std::abs(static_cast<double>(a) - static_cast<double>(b)) < (eps))

// Device-agnostic tensor comparison via to_vector<double>().
// For CUDA, to_vector does a single cudaMemcpy — no per-element round-trips.
inline bool tensors_match(const otter::Tensor& a, const otter::Tensor& b,
                           double tol = 1e-9) {
    if (a.shape() != b.shape()) return false;
    auto va = a.to_vector<double>();
    auto vb = b.to_vector<double>();
    for (std::size_t i = 0; i < va.size(); ++i)
        if (std::abs(va[i] - vb[i]) >= tol) return false;
    return true;
}

// Try a call; return true if it threw std::runtime_error whose what() contains fragment.
template<typename F>
inline bool throws_with(F&& f, const char* fragment, std::string& out_msg) {
    try { f(); return false; }
    catch (const std::runtime_error& e) {
        out_msg = e.what();
        return std::string(e.what()).find(fragment) != std::string::npos;
    }
    catch (...) { return false; }
}
