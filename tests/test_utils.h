#pragma once

#include <cmath>
#include <iostream>

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
