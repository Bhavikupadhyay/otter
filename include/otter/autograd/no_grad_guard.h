#pragma once

namespace otter {

// NoGradGuard — RAII guard that disables gradient tracking for its lifetime.
//
// While a NoGradGuard is alive, Operation::execute() skips all graph wiring
// (no grad_op_ set, no GradAccumulator allocated, no BroadcastOp nodes carry
// grad history) even for tensors with requires_grad=true.
//
// Usage:
//   {
//       otter::NoGradGuard ng;
//       Tensor out = model.forward(input);  // no grad graph built
//   }  // grad tracking restored here
//
// Thread-local: each thread has its own flag. A guard on one thread does not
// affect gradient tracking on other threads.
//
// Nesting: safe. Each guard saves the previous flag value and restores it on
// destruction, so nested guards correctly restore the outer guard's state.
struct NoGradGuard {
    NoGradGuard() noexcept  : prev_(grad_mode()) { grad_mode() = false; }
    ~NoGradGuard() noexcept { grad_mode() = prev_; }

    NoGradGuard(const NoGradGuard&)            = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;
    NoGradGuard(NoGradGuard&&)                 = delete;
    NoGradGuard& operator=(NoGradGuard&&)      = delete;

    // Returns a reference to the thread-local enabled flag.
    //   true  → gradient tracking is on (default)
    //   false → no graph wiring in execute()
    [[nodiscard]] static bool& grad_mode() noexcept {
        thread_local bool enabled = true;
        return enabled;
    }

private:
    bool prev_;
};

} // namespace otter
