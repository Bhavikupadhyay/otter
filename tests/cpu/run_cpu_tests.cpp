#include "../shared/test_shared_broadcast.h"
#include "../shared/test_shared_factories.h"
#include "../shared/test_shared_ops_backward.h"
#include "../shared/test_shared_ops_forward.h"
#include "../shared/test_shared_tensor.h"
#include "../shared/test_shared_thread_safety.h"
#include "../shared/test_shared_views.h"

#include "otter/backends/cpu.h"

// Forward declarations for CPU-specific runners defined in separate TUs.
namespace otter::test {
void run_cpu_memory_tests();
void run_cpu_kernel_impl_tests();
} // namespace otter::test

namespace otter::test {

void run_all_cpu_tests() {
    Backend& be = cpu_backend();

    // ── Shared tests (device-agnostic, parameterised over Backend&) ───────────
    shared::run_shared_factories(be);
    shared::run_shared_tensor(be);
    shared::run_shared_ops_forward(be);
    shared::run_shared_ops_backward(be);
    shared::run_shared_broadcast(be);
    shared::run_shared_views(be);
    shared::run_shared_thread_safety(be);

    // ── CPU-specific tests ────────────────────────────────────────────────────
    run_cpu_memory_tests();
    run_cpu_kernel_impl_tests();
}

} // namespace otter::test
