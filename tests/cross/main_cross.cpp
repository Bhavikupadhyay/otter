#include <iostream>
#include "../utils/test_utils.h"

namespace otter::test {
void run_device_transfer_tests();
} // namespace otter::test

int main() {
    otter::test::run_device_transfer_tests();

    std::cout << "\n"
              << otter::test::tests_passed << " / "
              << otter::test::tests_run    << " tests passed\n";

    return (otter::test::tests_passed == otter::test::tests_run) ? 0 : 1;
}
