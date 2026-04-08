#include "test_utils.h"

#include <iostream>

namespace otter::test {
void run_memory_tests();
void run_kernel_tests();
void run_cpu_kernel_tests();
void run_tensor_tests();
void run_autograd_tests();
void run_broadcast_tests();
void run_matmul_tests();
void run_views_tests();
void run_math_ops_tests();
} // namespace otter::test

int main() {
    otter::test::run_memory_tests();
    otter::test::run_kernel_tests();
    otter::test::run_cpu_kernel_tests();
    otter::test::run_tensor_tests();
    otter::test::run_autograd_tests();
    otter::test::run_broadcast_tests();
    otter::test::run_matmul_tests();
    otter::test::run_views_tests();
    otter::test::run_math_ops_tests();

    std::cout << "\n"
              << otter::test::tests_passed << " / "
              << otter::test::tests_run    << " passed\n";

    return (otter::test::tests_passed == otter::test::tests_run) ? 0 : 1;
}
