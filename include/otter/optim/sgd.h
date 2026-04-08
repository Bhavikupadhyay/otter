#pragma once

#include "otter/optim/optimizer.h"

namespace otter::optim {

// SGD — stochastic gradient descent with optional momentum and weight decay.
//
// Update rule (standard form):
//
//   Effective gradient:   g_eff = grad + weight_decay * param   (if wd != 0)
//                         g_eff = grad                          (otherwise)
//
//   With momentum:        v = momentum * v + g_eff
//                         param -= lr * v
//
//   Without momentum:     param -= lr * g_eff
//
// Per-parameter velocity buffers are allocated lazily on the first step() call
// where momentum != 0.0. They persist across steps as member state.
//
// Parameters whose .grad() is undefined are skipped silently.
class SGD final : public Optimizer {
public:
    SGD(std::vector<Tensor> params,
        double lr,
        double momentum     = 0.0,
        double weight_decay = 0.0);

    void step() override;

private:
    double              momentum_;
    double              weight_decay_;
    std::vector<Tensor> velocity_;  // parallel to params_; undefined until first step
};

} // namespace otter::optim
