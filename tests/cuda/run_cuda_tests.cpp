#include "../shared/test_shared_broadcast.h"
#include "../shared/test_shared_factories.h"
#include "../shared/test_shared_ops_backward.h"
#include "../shared/test_shared_ops_forward.h"
#include "../shared/test_shared_tensor.h"
#include "../shared/test_shared_thread_safety.h"
#include "../shared/test_shared_views.h"

#include "otter/backends/cuda.h"

// Forward declarations for CUDA-specific runners defined in separate TUs.
namespace otter::test {
void run_cuda_memory_tests();
void run_cuda_stream_tests();
void run_cuda_concurrency_tests();
} // namespace otter::test

namespace otter::test {

void run_all_cuda_tests() {
    Backend& be = cuda_backend();

    // ── Shared tests (device-agnostic, parameterised over Backend&) ───────────
    shared::run_shared_factories(be);
    shared::run_shared_tensor(be);
    shared::run_shared_ops_forward(be);
    shared::run_shared_ops_backward(be);
    shared::run_shared_broadcast(be);
    shared::run_shared_views(be);
    shared::run_shared_thread_safety(be);

    // ── CUDA-specific tests ───────────────────────────────────────────────────
    run_cuda_memory_tests();
    run_cuda_stream_tests();
    run_cuda_concurrency_tests();
}

} // namespace otter::test
