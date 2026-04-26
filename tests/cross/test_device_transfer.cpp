#include "../utils/test_utils.h"

#include <cstddef>
#include <vector>

#include "otter/backends/cpu.h"
#include "otter/backends/cuda.h"
#include "otter/tensor.h"

namespace otter::test {

void test_cuda_F1_cpu_to_cuda_to_cpu_round_trip() {
    std::cout << "[CUDA F1] CPU → cuda() → cpu() round-trip: values preserved\n";
    Tensor h  = Tensor::from_data<double>({1.1, 2.2, 3.3, 4.4}, {4}, cpu_backend());
    Tensor d  = h.cuda();
    Tensor h2 = d.cpu();
    CHECK(h2.backend().device() == Device::CPU);
    CHECK_NEAR(h2.at({0}), 1.1, 1e-10);
    CHECK_NEAR(h2.at({1}), 2.2, 1e-10);
    CHECK_NEAR(h2.at({2}), 3.3, 1e-10);
    CHECK_NEAR(h2.at({3}), 4.4, 1e-10);
}

void test_cuda_F2_2d_tensor_round_trip() {
    std::cout << "[CUDA F2] 2D {2,3} CPU → cuda → cpu: at({r,c}) preserved\n";
    std::vector<double> vals = {1,2,3,4,5,6};
    Tensor h  = Tensor::from_data<double>(vals, {2,3}, cpu_backend());
    Tensor d  = h.cuda();
    Tensor h2 = d.cpu();
    CHECK(h2.shape() == (std::vector<std::size_t>{2, 3}));
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            CHECK_NEAR(h2.at({r, c}), vals[r * 3 + c], 1e-12);
}

void test_cuda_F3_noncontiguous_cpu_to_cuda() {
    std::cout << "[CUDA F3] non-contiguous CPU tensor → cuda(): CPU contiguous() "
                 "runs first, values correct on CUDA\n";
    Tensor h  = Tensor::from_data<double>({1,2,3,4}, {2,2}, cpu_backend());
    Tensor ht = h.transpose(0, 1);
    CHECK(!ht.is_contiguous());

    Tensor d = ht.cuda();
    CHECK(d.backend().device() == Device::CUDA);
    CHECK(d.is_contiguous());
    CHECK_NEAR(d.at({0,0}), ht.at({0,0}), 1e-12);
    CHECK_NEAR(d.at({0,1}), ht.at({0,1}), 1e-12);
    CHECK_NEAR(d.at({1,0}), ht.at({1,0}), 1e-12);
    CHECK_NEAR(d.at({1,1}), ht.at({1,1}), 1e-12);
}

void test_cuda_F4_noncontiguous_cuda_to_cpu_works() {
    std::cout << "[CUDA F4] non-contiguous CUDA tensor → cpu(): "
                 "CUDA copy kernel materialises before transfer\n";
    Tensor t  = Tensor::from_data<double>({1,2,3,4,5,6}, {2,3}, cuda_backend());
    Tensor nc = t.transpose(0, 1);
    CHECK(!nc.is_contiguous());
    Tensor h = nc.cpu();
    CHECK(h.backend().device() == Device::CPU);
    CHECK(h.is_contiguous());
    CHECK(h.shape() == (std::vector<std::size_t>{3, 2}));
    CHECK_NEAR(h.at({0,0}), nc.at({0,0}), 1e-12);
    CHECK_NEAR(h.at({0,1}), nc.at({0,1}), 1e-12);
    CHECK_NEAR(h.at({1,0}), nc.at({1,0}), 1e-12);
    CHECK_NEAR(h.at({1,1}), nc.at({1,1}), 1e-12);
    CHECK_NEAR(h.at({2,0}), nc.at({2,0}), 1e-12);
    CHECK_NEAR(h.at({2,1}), nc.at({2,1}), 1e-12);
}

void test_cuda_F5_to_cuda_on_cuda_is_fast_path() {
    std::cout << "[CUDA F5] to(Device::CUDA) on CUDA tensor is the identity fast path\n";
    const std::size_t before   = cuda_backend().memory_manager()->bytes_allocated();
    Tensor t = Tensor::zeros({8}, cuda_backend());
    t.fill_(5.0);
    const std::size_t with_t   = cuda_backend().memory_manager()->bytes_allocated();
    Tensor same = t.to(Device::CUDA);
    CHECK(cuda_backend().memory_manager()->bytes_allocated() == with_t);
    CHECK_NEAR(same.at({0}), 5.0, 1e-12);
    (void)before;
}

void test_cuda_F6_to_cpu_on_cpu_is_fast_path() {
    std::cout << "[CUDA F6] to(Device::CPU) on CPU tensor is the identity fast path\n";
    Tensor t    = Tensor::from_data<double>({1,2,3}, {3}, cpu_backend());
    Tensor same = t.to(Device::CPU);
    CHECK(same.backend().device() == Device::CPU);
    CHECK_NEAR(same.at({0}), 1.0, 1e-12);
}

void test_cuda_F7_to_does_not_preserve_requires_grad() {
    std::cout << "[CUDA F7] to() drops autograd history — returned tensor is a plain leaf\n";
    Tensor h = Tensor::from_data<double>({1,2,3}, {3}, cpu_backend(),
                                         /*requires_grad=*/true);
    CHECK(h.requires_grad());
    Tensor d = h.cuda();
    CHECK(!d.requires_grad());
    Tensor h2 = d.cpu();
    CHECK(!h2.requires_grad());
}

void test_cuda_F8_bytes_allocated_restored_after_transfer() {
    std::cout << "[CUDA F8] CUDA bytes_allocated returns to baseline after transfer "
                 "tensor is destroyed\n";
    const std::size_t base = cuda_backend().memory_manager()->bytes_allocated();
    {
        Tensor h = Tensor::from_data<double>({1,2,3,4}, {4}, cpu_backend());
        Tensor d = h.cuda();
        CHECK(cuda_backend().memory_manager()->bytes_allocated() > base);
    }
    CHECK(cuda_backend().memory_manager()->bytes_allocated() == base);
}

void test_cuda_F9_large_tensor_transfer() {
    std::cout << "[CUDA F9] 512-element: CPU fill → cuda() → cpu() spot-check\n";
    constexpr std::size_t N = 512;
    std::vector<double> vals(N);
    for (std::size_t i = 0; i < N; ++i) vals[i] = static_cast<double>(i) * 0.5;
    Tensor h  = Tensor::from_data<double>(vals, {N}, cpu_backend());
    Tensor d  = h.cuda();
    Tensor h2 = d.cpu();
    CHECK_NEAR(h2.at({0}),     vals[0],     1e-10);
    CHECK_NEAR(h2.at({N/4}),   vals[N/4],   1e-10);
    CHECK_NEAR(h2.at({N/2}),   vals[N/2],   1e-10);
    CHECK_NEAR(h2.at({N - 1}), vals[N - 1], 1e-10);
}

void run_device_transfer_tests() {
    test_cuda_F1_cpu_to_cuda_to_cpu_round_trip();
    test_cuda_F2_2d_tensor_round_trip();
    test_cuda_F3_noncontiguous_cpu_to_cuda();
    test_cuda_F4_noncontiguous_cuda_to_cpu_works();
    test_cuda_F5_to_cuda_on_cuda_is_fast_path();
    test_cuda_F6_to_cpu_on_cpu_is_fast_path();
    test_cuda_F7_to_does_not_preserve_requires_grad();
    test_cuda_F8_bytes_allocated_restored_after_transfer();
    test_cuda_F9_large_tensor_transfer();
}

} // namespace otter::test
