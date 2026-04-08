#include "otter/optim/optimizer.h"

namespace otter::optim {

Optimizer::Optimizer(std::vector<Tensor> params, double lr)
    : params_(std::move(params))
    , lr_(lr)
{}

void Optimizer::zero_grad() {
    for (Tensor& p : params_)
        p.zero_grad();
}

} // namespace otter::optim
