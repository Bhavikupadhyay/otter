#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "otter/ops/operation.h"  // transitively provides Tensor, GradAccumulator

namespace otter {

// GraphExecutor — singleton thread pool that drives the autograd backward pass.
//
// Owns a fixed-size pool of worker threads and a shared ready-queue. run() owns
// Phases 1–5 of the backward pass (DFS, dep-count setup, grad clearing, seed,
// parallel traversal, cleanup).
//
// Correctness model:
//   dep_counts_[op] = number of ops whose backward() will still send a gradient
//   contribution to op. When it reaches zero, all contributions have arrived and
//   op is safe to process. An atomic fetch_sub with acq_rel semantics establishes
//   happens-before between the last gradient write and the enqueue that follows.
//
// One backward pass at a time: run_mtx_ serializes concurrent run() calls.
// This is correct for single-model training loops. Multi-pass concurrency is a
// future concern (requires per-pass context structs).
//
// The dep-count model provides correctness for concurrent CUDA backward passes.
class GraphExecutor {
public:
    [[nodiscard]] static GraphExecutor& instance();

    // Full backward pass: Phases 1–5. Called by Tensor::backward().
    void run(Tensor& root, Tensor seed, bool retain_graph);

    GraphExecutor(const GraphExecutor&)            = delete;
    GraphExecutor& operator=(const GraphExecutor&) = delete;
    GraphExecutor(GraphExecutor&&)                 = delete;
    GraphExecutor& operator=(GraphExecutor&&)      = delete;

private:
    explicit GraphExecutor(std::size_t num_threads = 4);
    ~GraphExecutor();

    void worker_loop();
    void process_node(Tensor node);

    // ── Thread pool ────────────────────────────────────────────────────────────
    std::vector<std::thread> workers_;

    // ── Per-backward-call state (valid only while run_mtx_ is held) ───────────
    // dep_counts_: atomic because multiple workers decrement concurrently.
    // unordered_map structural modifications (insert/erase) happen only in run()
    // before workers are notified — no structural race.
    std::mutex run_mtx_;
    std::unordered_map<ops::Operation*, std::atomic<int>> dep_counts_;

    // ── Ready queue ────────────────────────────────────────────────────────────
    std::deque<Tensor>      ready_queue_;
    std::mutex              queue_mtx_;
    std::condition_variable queue_cv_;

    // ── Completion tracking ────────────────────────────────────────────────────
    // pending_ = nodes currently in the queue OR being processed by a worker.
    // Incremented before enqueueing a newly-ready node (before decrementing for
    // the current node) to prevent a spurious zero crossing.
    std::atomic<int>        pending_{0};
    std::condition_variable done_cv_;

    bool stop_ = false;
};

} // namespace otter
