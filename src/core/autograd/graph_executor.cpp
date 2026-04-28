#include "core/autograd/graph_executor.h"

#include <algorithm>
#include <cassert>
#include <unordered_set>

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
// Blocks on queue_cv_ until a node is available or stop_ is set.
// Each popped node is processed by process_node(), which decrements pending_
// and notifies done_cv_ if the count reaches zero.

void GraphExecutor::worker_loop() {
    while (true) {
        Tensor node;
        {
            std::unique_lock<std::mutex> lock(queue_mtx_);
            queue_cv_.wait(lock, [this] { return stop_ || !ready_queue_.empty(); });
            if (stop_ && ready_queue_.empty()) break;
            if (ready_queue_.empty()) continue;  // spurious wakeup
            node = std::move(ready_queue_.front());
            ready_queue_.pop_front();
        }
        process_node(std::move(node));
    }
}

// ── process_node ──────────────────────────────────────────────────────────────
//
// Executes one backward node:
//   1. Read accumulated gradient (under GradAccumulator::mtx).
//   2. Snapshot saved inputs (under saved_mtx_).
//   3. Call op->backward().
//   4. For each saved input: accumulate_grad(), then fetch_sub dep_count.
//      If dep_count hits zero: increment pending_, enqueue, notify a worker.
//   5. fetch_sub pending_ for this node. If it hits zero: notify done_cv_.
//
// pending_ management: new nodes are incremented BEFORE this node's decrement to
// prevent a spurious zero when the current node unblocks at least one successor.

void GraphExecutor::process_node(Tensor node) {
    // ── Early exits all decrement pending_ and notify if zero ─────────────────
    auto finish = [this] {
        if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1)
            done_cv_.notify_one();
    };

    if (!node.grad_op_) { finish(); return; }

    // Read accumulated gradient under lock.
    Tensor node_grad;
    {
        auto acc = std::atomic_load(&node.grad_accum_);
        if (!acc) { finish(); return; }
        std::lock_guard<std::mutex> lock(acc->mtx);
        if (!acc->grad_tensor.defined()) { finish(); return; }
        node_grad = acc->grad_tensor;  // by-value copy under lock
    }

    const std::vector<Tensor> saved = node.grad_op_->snapshot_inputs();
    if (saved.empty()) { finish(); return; }

    auto input_grads = node.grad_op_->backward({node_grad});

    // Accumulate, then decrement predecessor dep_counts.
    for (std::size_t i = 0; i < saved.size() && i < input_grads.size(); ++i) {
        if (!saved[i].requires_grad() || !input_grads[i].defined()) continue;

        saved[i].accumulate_grad(input_grads[i]);

        if (!saved[i].grad_op_) continue;
        auto it = dep_counts_.find(saved[i].grad_op_.get());
        if (it == dep_counts_.end()) continue;

        // fetch_sub returns the old value. If it was 1, dep_count just hit 0.
        // Increment pending_ for the new node BEFORE decrementing for self so
        // pending_ never crosses zero while work remains.
        if (it->second.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            pending_.fetch_add(1, std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> qlock(queue_mtx_);
                ready_queue_.push_back(saved[i]);
            }
            queue_cv_.notify_one();
        }
    }

    finish();
}

// ── run ───────────────────────────────────────────────────────────────────────
//
// Entry point from Tensor::backward(). Runs one complete backward pass:
//
//   Phase 1: topological DFS → order vector
//   Phase 2: clear intermediate gradients
//   Phase 3: seed root gradient
//   Phase 4: build dep_counts, seed ready_queue, dispatch to workers, wait
//   Phase 5: cleanup (clear_saved, null root grad_op_)
//
// run_mtx_ ensures one pass at a time.

void GraphExecutor::run(Tensor& root, Tensor seed, bool retain_graph) {
    std::lock_guard<std::mutex> run_lock(run_mtx_);

    // ── Phase 1: topological DFS ──────────────────────────────────────────────
    std::vector<Tensor>                  order;
    std::unordered_set<ops::Operation*>  visited;
    Tensor::topo_dfs(root, visited, order);
    std::reverse(order.begin(), order.end());

    // ── Phase 2: clear intermediate gradients ─────────────────────────────────
    // Single-threaded; workers are not yet running. No race.
    for (const Tensor& node : order) {
        auto acc = std::atomic_load(&node.grad_accum_);
        if (!node.is_leaf_ && acc) {
            std::lock_guard<std::mutex> lock(acc->mtx);
            acc->grad_tensor = Tensor{};
        }
    }

    // ── Phase 3: seed root gradient ───────────────────────────────────────────
    {
        // get_or_create: atomically load; if null, CAS in a fresh accumulator.
        std::shared_ptr<GradAccumulator> acc = std::atomic_load(&root.grad_accum_);
        if (!acc) {
            auto fresh = std::make_shared<GradAccumulator>();
            if (!std::atomic_compare_exchange_strong(&root.grad_accum_, &acc, fresh))
                acc = std::atomic_load(&root.grad_accum_);
            else
                acc = fresh;
        }
        std::lock_guard<std::mutex> lock(acc->mtx);
        acc->grad_tensor = std::move(seed);
    }

    // ── Phase 4: build dep_counts, seed ready_queue, dispatch ─────────────────
    //
    // Two-pass to guarantee all entries exist before incrementing:
    //   Pass 1 — emplace every op with count 0.
    //   Pass 2 — increment for each (op → saved_input's grad_op_) edge.
    dep_counts_.clear();
    dep_counts_.reserve(order.size());

    for (const Tensor& node : order) {
        if (!node.grad_op_) continue;
        dep_counts_.try_emplace(node.grad_op_.get(), 0);
    }
    for (const Tensor& node : order) {
        if (!node.grad_op_) continue;
        for (const Tensor& inp : node.grad_op_->inputs()) {
            if (inp.requires_grad() && inp.grad_op_) {
                auto it = dep_counts_.find(inp.grad_op_.get());
                if (it != dep_counts_.end())
                    it->second.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    // Seed the ready queue with ops whose dep_count is 0 (the root op).
    {
        std::lock_guard<std::mutex> qlock(queue_mtx_);
        assert(ready_queue_.empty() && "ready_queue_ must be empty at start of run()");
        for (const Tensor& node : order) {
            if (node.grad_op_ &&
                dep_counts_[node.grad_op_.get()].load(std::memory_order_relaxed) == 0)
                ready_queue_.push_back(node);
        }
        pending_.store(static_cast<int>(ready_queue_.size()),
                       std::memory_order_relaxed);
    }

    if (pending_.load(std::memory_order_relaxed) > 0)
        queue_cv_.notify_all();

    // Wait for workers to drain the queue.
    {
        std::unique_lock<std::mutex> qlock(queue_mtx_);
        done_cv_.wait(qlock, [this] { return pending_.load(std::memory_order_acquire) == 0; });
    }

    // ── Phase 5: cleanup ──────────────────────────────────────────────────────
    if (!retain_graph) {
        for (const Tensor& node : order)
            if (node.grad_op_) node.grad_op_->clear_saved();
        root.grad_op_.reset();
    }
}

} // namespace otter
