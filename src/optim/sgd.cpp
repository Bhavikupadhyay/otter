#include "otter/optim/sgd.h"

#include <cassert>

#include "otter/autograd/no_grad_guard.h"
#include "otter/kernel/kernel_engine.h"

namespace otter::optim {

SGD::SGD(std::vector<Tensor> params,
         double lr,
         double momentum,
         double weight_decay)
    : Optimizer(std::move(params), lr)
    , momentum_(momentum)
    , weight_decay_(weight_decay)
    , velocity_(params_.size())   // default-constructed (undefined) Tensors
{}

void SGD::step() {
    NoGradGuard ng;

    for (std::size_t i = 0; i < params_.size(); ++i) {
        Tensor grad = params_[i].grad();
        if (!grad.defined()) continue;

        auto* ke = params_[i].backend().kernel_engine();

        // ── Build effective gradient ──────────────────────────────────────────
        // If weight_decay != 0, g_eff = grad + wd * param (needs a scratch copy).
        // Otherwise g shares the grad Buffer — read-only below, no allocation.
        Tensor g;
        if (weight_decay_ != 0.0) {
            g = Tensor::zeros_like(params_[i]);
            ke->dispatch_axpy(g, 1.0,           grad);        // g  = grad
            ke->dispatch_axpy(g, weight_decay_, params_[i]);  // g += wd * w
        } else {
            g = grad;  // value-type copy, shares Buffer
        }

        // ── Apply update ──────────────────────────────────────────────────────
        if (momentum_ != 0.0) {
            if (!velocity_[i].defined())
                velocity_[i] = Tensor::zeros_like(params_[i]);
            ke->dispatch_scale(velocity_[i], momentum_);           // v  = m * v
            ke->dispatch_axpy (velocity_[i], 1.0, g);              // v += g
            ke->dispatch_axpy (params_[i],  -lr_, velocity_[i]);   // w -= lr * v
        } else {
            ke->dispatch_axpy(params_[i], -lr_, g);                // w -= lr * g
        }
    }
}

} // namespace otter::optim
