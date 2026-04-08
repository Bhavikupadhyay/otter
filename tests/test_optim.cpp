#include "test_utils.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "otter/tensor.h"
#include "otter/backends/cpu.h"
#include "otter/optim/sgd.h"

// Tests for dispatch_scale, dispatch_axpy, and SGD optimizer.
// Covers: kernel unit tests, optimizer unit tests, and a full 2-layer
// MLP training loop demonstrating end-to-end gradient-based learning.

namespace otter::test {

// ── dispatch_scale ────────────────────────────────────────────────────────────

void test_scale_contiguous() {
    std::cout << "[Optim 1] dispatch_scale: contiguous tensor scaled correctly\n";
    Backend& be = cpu_backend();
    Tensor x = Tensor::from_data<double>({1.0, 2.0, 3.0, 4.0}, {4}, be);
    be.kernel_engine()->dispatch_scale(x, 3.0);
    CHECK_NEAR(x.at({0}), 3.0,  1e-12);
    CHECK_NEAR(x.at({1}), 6.0,  1e-12);
    CHECK_NEAR(x.at({2}), 9.0,  1e-12);
    CHECK_NEAR(x.at({3}), 12.0, 1e-12);
}

void test_scale_zero() {
    std::cout << "[Optim 2] dispatch_scale: scale by 0.0 zeros the tensor\n";
    Backend& be = cpu_backend();
    Tensor x = Tensor::from_data<double>({5.0, -3.0, 2.0}, {3}, be);
    be.kernel_engine()->dispatch_scale(x, 0.0);
    for (std::size_t i = 0; i < 3; ++i)
        CHECK_NEAR(x.at({i}), 0.0, 1e-12);
}

// ── dispatch_axpy ─────────────────────────────────────────────────────────────

void test_axpy_contiguous() {
    std::cout << "[Optim 3] dispatch_axpy: dst += alpha*src on contiguous tensors\n";
    Backend& be = cpu_backend();
    // dst = [1, 2, 3],  src = [4, 5, 6],  alpha = 2.0
    // expected: [1+8, 2+10, 3+12] = [9, 12, 15]
    Tensor dst = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be);
    Tensor src = Tensor::from_data<double>({4.0, 5.0, 6.0}, {3}, be);
    be.kernel_engine()->dispatch_axpy(dst, 2.0, src);
    CHECK_NEAR(dst.at({0}), 9.0,  1e-12);
    CHECK_NEAR(dst.at({1}), 12.0, 1e-12);
    CHECK_NEAR(dst.at({2}), 15.0, 1e-12);
}

void test_axpy_negative_alpha() {
    std::cout << "[Optim 4] dispatch_axpy: alpha=-1.0 subtracts src from dst\n";
    Backend& be = cpu_backend();
    Tensor dst = Tensor::from_data<double>({10.0, 10.0, 10.0}, {3}, be);
    Tensor src = Tensor::from_data<double>({1.0,  2.0,  3.0},  {3}, be);
    be.kernel_engine()->dispatch_axpy(dst, -1.0, src);
    CHECK_NEAR(dst.at({0}), 9.0, 1e-12);
    CHECK_NEAR(dst.at({1}), 8.0, 1e-12);
    CHECK_NEAR(dst.at({2}), 7.0, 1e-12);
}

void test_axpy_strided() {
    std::cout << "[Optim 5] dispatch_axpy: strided dst (transposed 2x2)\n";
    Backend& be = cpu_backend();
    // base = [[1,2],[3,4]], transposed view has stride [1,2] instead of [2,1]
    Tensor base = Tensor::from_data<double>({1.0, 2.0, 3.0, 4.0}, {2, 2}, be);
    Tensor dst  = base.transpose(0, 1);  // strided view of base's buffer
    Tensor src  = Tensor::from_data<double>({10.0, 20.0, 30.0, 40.0}, {2, 2}, be);
    // dst logical layout before: [[1,3],[2,4]]
    // src logical layout:        [[10,20],[30,40]]
    // after axpy(dst, 1.0, src): dst[i,j] += src[i,j]
    //   [0,0]: 1+10=11  [0,1]: 3+20=23  [1,0]: 2+30=32  [1,1]: 4+40=44
    be.kernel_engine()->dispatch_axpy(dst, 1.0, src);
    CHECK_NEAR(dst.at({0, 0}), 11.0, 1e-12);
    CHECK_NEAR(dst.at({0, 1}), 23.0, 1e-12);
    CHECK_NEAR(dst.at({1, 0}), 32.0, 1e-12);
    CHECK_NEAR(dst.at({1, 1}), 44.0, 1e-12);
}

// ── SGD unit tests ────────────────────────────────────────────────────────────

void test_sgd_basic_step() {
    std::cout << "[Optim 6] SGD: basic step without momentum — param -= lr*grad\n";
    Backend& be = cpu_backend();
    // w = [2, 4, 6],  grad = [1, 1, 1],  lr = 0.1
    // expected: [1.9, 3.9, 5.9]
    Tensor w = Tensor::from_data<double>({2.0, 4.0, 6.0}, {3}, be,
                                         /*requires_grad=*/true);
    // Manually wire a gradient (simulate what backward() would produce).
    Tensor g = Tensor::from_data<double>({1.0, 1.0, 1.0}, {3}, be);
    w.accumulate_grad(g);

    optim::SGD sgd({w}, /*lr=*/0.1);
    sgd.step();

    CHECK_NEAR(w.at({0}), 1.9, 1e-12);
    CHECK_NEAR(w.at({1}), 3.9, 1e-12);
    CHECK_NEAR(w.at({2}), 5.9, 1e-12);
}

void test_sgd_zero_grad() {
    std::cout << "[Optim 7] SGD: zero_grad clears accumulated gradients\n";
    Backend& be = cpu_backend();
    Tensor w = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be,
                                         /*requires_grad=*/true);
    Tensor g = Tensor::from_data<double>({5.0, 5.0, 5.0}, {3}, be);
    w.accumulate_grad(g);
    CHECK(w.grad().defined());

    optim::SGD sgd({w}, /*lr=*/0.1);
    sgd.zero_grad();
    CHECK(!w.grad().defined());
}

void test_sgd_no_grad_no_update() {
    std::cout << "[Optim 8] SGD: param unchanged when grad is undefined\n";
    Backend& be = cpu_backend();
    Tensor w = Tensor::from_data<double>({3.0, 3.0}, {2}, be,
                                         /*requires_grad=*/true);
    // No accumulate_grad call — grad stays undefined.
    optim::SGD sgd({w}, /*lr=*/0.1);
    sgd.step();
    CHECK_NEAR(w.at({0}), 3.0, 1e-12);
    CHECK_NEAR(w.at({1}), 3.0, 1e-12);
}

void test_sgd_momentum() {
    std::cout << "[Optim 9] SGD: momentum — velocity accumulates across steps\n";
    Backend& be = cpu_backend();
    // w = [0],  grad = [1] every step,  lr=1.0,  momentum=0.9
    // Step 1: v=0*0.9+1=1,   w = 0 - 1*1 = -1
    // Step 2: v=1*0.9+1=1.9, w = -1 - 1*1.9 = -2.9
    // Step 3: v=1.9*0.9+1=2.71, w = -2.9 - 1*2.71 = -5.61
    Tensor w = Tensor::from_data<double>({0.0}, {1}, be, /*requires_grad=*/true);
    optim::SGD sgd({w}, /*lr=*/1.0, /*momentum=*/0.9);

    auto step_with_grad = [&](double grad_val) {
        w.zero_grad();
        Tensor g = Tensor::from_data<double>({grad_val}, {1}, be);
        w.accumulate_grad(g);
        sgd.step();
    };

    step_with_grad(1.0);
    CHECK_NEAR(w.at({0}), -1.0, 1e-10);

    step_with_grad(1.0);
    CHECK_NEAR(w.at({0}), -2.9, 1e-10);

    step_with_grad(1.0);
    CHECK_NEAR(w.at({0}), -5.61, 1e-10);
}

void test_sgd_weight_decay() {
    std::cout << "[Optim 10] SGD: weight_decay — g_eff = grad + wd*w\n";
    Backend& be = cpu_backend();
    // w=2, grad=1, wd=0.5, lr=1.0 → g_eff=1+0.5*2=2 → w=2-1*2=0
    Tensor w = Tensor::from_data<double>({2.0}, {1}, be, /*requires_grad=*/true);
    Tensor g = Tensor::from_data<double>({1.0}, {1}, be);
    w.accumulate_grad(g);

    optim::SGD sgd({w}, /*lr=*/1.0, /*momentum=*/0.0, /*weight_decay=*/0.5);
    sgd.step();
    CHECK_NEAR(w.at({0}), 0.0, 1e-12);
}

void test_sgd_set_lr() {
    std::cout << "[Optim 11] SGD: set_lr updates learning rate for next step\n";
    Backend& be = cpu_backend();
    Tensor w = Tensor::from_data<double>({10.0}, {1}, be, /*requires_grad=*/true);
    Tensor g = Tensor::from_data<double>({1.0},  {1}, be);
    w.accumulate_grad(g);

    optim::SGD sgd({w}, /*lr=*/0.1);
    sgd.set_lr(2.0);
    sgd.step();
    // w = 10 - 2.0*1 = 8
    CHECK_NEAR(w.at({0}), 8.0, 1e-12);
}

// ── Mini MLP training loop ────────────────────────────────────────────────────
//
// 2-layer fully connected network trained on a fixed dataset with MSE loss.
// Architecture:
//   input  {4, 2}
//   Layer1: W1 {2, 4} + b1 {4}, relu activation  → hidden {4, 4}
//   Layer2: W2 {4, 1} + b2 {1}                   → output {4, 1}
//   Loss: MSE = mean((output - target)^2)
//
// Weights are initialised to small non-zero constants so that relu is not
// all-zero on the first pass and gradients flow from the start.
//
// Assert: loss strictly decreases on every one of 30 SGD steps.

static Tensor mlp_forward(const Tensor& x,
                           Tensor& W1, Tensor& b1,
                           Tensor& W2, Tensor& b2)
{
    // hidden = relu(x @ W1 + broadcast(b1))
    Tensor pre1 = x.matmul(W1);                             // {4, 4}
    Tensor h    = pre1.add(b1.broadcast_to({4, 4})).relu(); // {4, 4}

    // out = h @ W2 + broadcast(b2)
    Tensor pre2 = h.matmul(W2);                             // {4, 1}
    Tensor out  = pre2.add(b2.broadcast_to({4, 1}));        // {4, 1}
    return out;
}

void test_mlp_training_loop() {
    std::cout << "[Optim 12] MLP training loop: loss decreases for 30 SGD steps\n";
    Backend& be = cpu_backend();

    // ── Fixed dataset ─────────────────────────────────────────────────────────
    // 4 samples, 2 features; targets are a simple linear pattern.
    Tensor x = Tensor::from_data<double>(
        {0.1, 0.2,
         0.3, 0.4,
         0.5, 0.6,
         0.7, 0.8},
        {4, 2}, be);

    Tensor y = Tensor::from_data<double>(
        {0.3,
         0.7,
         1.1,
         1.5},
        {4, 1}, be);

    // ── Weights — small positive constants so relu is not dead at step 0 ─────
    Tensor W1 = Tensor::full({2, 4}, 0.1, be, DType::Float64, /*requires_grad=*/true);
    Tensor b1 = Tensor::full({4},    0.1, be, DType::Float64, /*requires_grad=*/true);
    Tensor W2 = Tensor::full({4, 1}, 0.1, be, DType::Float64, /*requires_grad=*/true);
    Tensor b2 = Tensor::full({1},    0.0, be, DType::Float64, /*requires_grad=*/true);

    optim::SGD sgd({W1, b1, W2, b2}, /*lr=*/0.01);

    double prev_loss = 1e18;
    for (int step = 0; step < 30; ++step) {
        sgd.zero_grad();

        Tensor out  = mlp_forward(x, W1, b1, W2, b2);
        Tensor diff = out.sub(y);
        Tensor loss = diff.mul(diff).mean();

        loss.backward();
        sgd.step();

        double cur_loss = loss.at({0});
        CHECK(cur_loss < prev_loss);
        prev_loss = cur_loss;
    }
}

// ── entry point ───────────────────────────────────────────────────────────────

void run_optim_tests() {
    test_scale_contiguous();
    test_scale_zero();
    test_axpy_contiguous();
    test_axpy_negative_alpha();
    test_axpy_strided();
    test_sgd_basic_step();
    test_sgd_zero_grad();
    test_sgd_no_grad_no_update();
    test_sgd_momentum();
    test_sgd_weight_decay();
    test_sgd_set_lr();
    test_mlp_training_loop();
}

} // namespace otter::test
