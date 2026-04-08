#pragma once

#include <vector>

#include "otter/tensor.h"

namespace otter::optim {

// Optimizer — abstract base for all parameter update rules.
//
// Holds shallow copies of the user's parameter tensors. Because Tensor is
// value-type with shared Buffer and shared GradAccumulator, mutations via
// dispatch_axpy / dispatch_scale are immediately visible through the user's
// original handles, and zero_grad() clears the same accumulators the user sees.
//
// Optimizers are NOT Operations — they do not participate in the autograd
// graph, save inputs, or implement backward(). They read .grad from leaf
// nodes and mutate parameter buffers in-place, always under NoGradGuard.
class Optimizer {
public:
    explicit Optimizer(std::vector<Tensor> params, double lr);
    virtual ~Optimizer() = default;

    // Non-copyable, non-movable — owns per-parameter state (e.g., velocity).
    Optimizer(const Optimizer&)             = delete;
    Optimizer& operator=(const Optimizer&)  = delete;
    Optimizer(Optimizer&&)                  = delete;
    Optimizer& operator=(Optimizer&&)       = delete;

    // Clears .grad on every parameter. Call before each forward pass.
    void zero_grad();

    // Applies the update rule using accumulated .grad on each parameter.
    virtual void step() = 0;

    [[nodiscard]] double lr() const noexcept { return lr_; }

    // Allows LR scheduling: call set_lr() between steps.
    void set_lr(double lr) noexcept { lr_ = lr; }

protected:
    std::vector<Tensor> params_;
    double              lr_;
};

} // namespace otter::optim
