#include "test_utils.h"

#include <iostream>

namespace otter::test {
void run_memory_tests();
void run_kernel_tests();
void run_cpu_kernel_tests();
void run_tensor_tests();
} // namespace otter::test

int main() {
    otter::test::run_memory_tests();
    otter::test::run_kernel_tests();
    otter::test::run_cpu_kernel_tests();
    otter::test::run_tensor_tests();

    std::cout << "\n"
              << otter::test::tests_passed << " / "
              << otter::test::tests_run    << " passed\n";

    return (otter::test::tests_passed == otter::test::tests_run) ? 0 : 1;
}
