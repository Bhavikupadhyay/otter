#include "core/autograd/graph_executor.h"

#include "otter/tensor.h"

namespace otter {

// ── Singleton ─────────────────────────────────────────────────────────────────

GraphExecutor& GraphExecutor::instance() {
    static GraphExecutor inst;
    return inst;
}

// ── Constructor / destructor ──────────────────────────────────────────────────

GraphExecutor::GraphExecutor(std::size_t num_threads) {
    workers_.reserve(num_threads);
    for (std::size_t i = 0; i < num_threads; ++i)
        workers_.emplace_back([this] { worker_loop(); });
}

GraphExecutor::~GraphExecutor() {
    {
        std::lock_guard<std::mutex> lock(queue_mtx_);
        stop_ = true;
    }
    queue_cv_.notify_all();
    for (auto& t : workers_) t.join();
}

// ── Worker loop ───────────────────────────────────────────────────────────────
//
// Step 1: workers exist but never receive work — they wait until stop_ is set.
// Step 3: workers will pop ops from ready_queue_, call backward(), and manage
// dep_counts. The pop path is wired in at that step.

void GraphExecutor::worker_loop() {
    while (true) {
        std::unique_lock<std::mutex> lock(queue_mtx_);
        queue_cv_.wait(lock, [this] { return stop_ || !ready_queue_.empty(); });
        if (stop_ && ready_queue_.empty()) break;
        // Step 3 fills in the task-processing body here.
    }
}

// ── run ───────────────────────────────────────────────────────────────────────
//
// Step 1: sequentially executes the backward pass on the calling thread via
// Tensor::backward_impl(). The thread pool is idle throughout.

void GraphExecutor::run(Tensor& root, Tensor seed, bool retain_graph) {
    Tensor::backward_impl(root, std::move(seed), retain_graph);
}

} // namespace otter
