#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "otter/ops/operation.h"  // transitively provides Tensor

namespace otter {

// GraphExecutor — singleton thread pool that drives the autograd backward pass.
//
// Owns a fixed-size pool of worker threads and a shared ready-queue. The calling
// thread enqueues initial ready operations (dep_count == 0), then blocks until
// all work drains. Workers pop ops, run backward(), accumulate gradients into
// predecessor nodes, and re-enqueue predecessors whose dep_count reaches zero.
//
// §4.1 Step 1: run() delegates to Tensor::backward_impl(), which executes the
// backward graph sequentially on the calling thread. Workers exist but are idle
// — they become active in Step 3 when run() enqueues to the shared queue.
//
// Singleton lifetime: created on first call to instance(), destroyed at program
// exit. Worker threads are joined in the destructor.
class GraphExecutor {
public:
    [[nodiscard]] static GraphExecutor& instance();

    // Entry point called by Tensor::backward().
    // In Step 1: executes backward_impl() synchronously on the calling thread.
    // In Step 3+: enqueues initial ready ops and waits for worker completion.
    void run(Tensor& root, Tensor seed, bool retain_graph);

    GraphExecutor(const GraphExecutor&)            = delete;
    GraphExecutor& operator=(const GraphExecutor&) = delete;
    GraphExecutor(GraphExecutor&&)                 = delete;
    GraphExecutor& operator=(GraphExecutor&&)      = delete;

private:
    explicit GraphExecutor(std::size_t num_threads = 4);
    ~GraphExecutor();

    void worker_loop();

    std::vector<std::thread>                    workers_;
    std::deque<std::shared_ptr<ops::Operation>> ready_queue_;
    std::mutex                                  queue_mtx_;
    std::condition_variable                     queue_cv_;
    std::atomic<int>                            pending_{0};
    std::condition_variable                     done_cv_;
    bool                                        stop_ = false;
};

} // namespace otter
