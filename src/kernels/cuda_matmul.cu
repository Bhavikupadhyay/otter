// Must be first — provides raw_const<T> / raw_mutable<T> template bodies.
#include "dispatcher.h"
#include "cuda_kernel_engine.h"

#include "otter/tensor.h"
#include "otter/detail/stride_utils.h"

#include <cassert>
#include <cstddef>

#include <cuda_runtime.h>

namespace {

// Naive triple-loop matmul kernel. Contiguous inputs only (caller asserts).
//
// Each thread computes one output element out[batch, i, j].
// Batch strides are the flat-element counts per batch slice for each tensor:
//   a  : M * K elements per batch slice
//   b  : K * N elements per batch slice
//   out: M * N elements per batch slice
__global__ void matmul_kernel(const double* __restrict__ a,
                               const double* __restrict__ b,
                               double*       __restrict__ out,
                               std::size_t M, std::size_t N, std::size_t K,
                               std::size_t bs_a, std::size_t bs_b, std::size_t bs_out,
                               std::size_t total) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const std::size_t mn    = M * N;
    const std::size_t batch = idx / mn;
    const std::size_t rem   = idx % mn;
    const std::size_t i     = rem / N;
    const std::size_t j     = rem % N;

    double s = 0.0;
    for (std::size_t k = 0; k < K; ++k)
        s += a[batch * bs_a + i * K + k] * b[batch * bs_b + k * N + j];

    out[batch * bs_out + i * N + j] = s;
}

} // namespace

namespace otter {

void CUDAKernelEngine::cuda_matmul(const Tensor& a, const Tensor& b, Tensor& out) const {
    assert(a.is_contiguous() && b.is_contiguous() && out.is_contiguous() &&
           "cuda_matmul: all inputs must be contiguous");
    assert(out.shape().size() >= 2 &&
           "cuda_matmul: output tensor must be at least 2-dimensional");

    const auto& out_shape = out.shape();
    const std::size_t ndim = out_shape.size();
    const std::size_t M    = out_shape[ndim - 2];
    const std::size_t N    = out_shape[ndim - 1];
    const std::size_t K    = a.shape()[ndim - 1];

    std::size_t batch = 1;
    for (std::size_t d = 0; d + 2 < ndim; ++d) batch *= out_shape[d];

    const std::size_t total  = batch * M * N;
    const std::size_t bs_a   = M * K;
    const std::size_t bs_b   = K * N;
    const std::size_t bs_out = M * N;

    const double* lhs = raw_const<double>(a.buffer())             + a.offset();
    const double* rhs = raw_const<double>(b.buffer())             + b.offset();
    double*       res = raw_mutable<double>(out.mutable_buffer()) + out.offset();

    const int block = 256;
    const int grid  = static_cast<int>(
        (total + static_cast<std::size_t>(block) - 1) / static_cast<std::size_t>(block));
    matmul_kernel<<<grid, block>>>(lhs, rhs, res, M, N, K, bs_a, bs_b, bs_out, total);

    cudaError_t err = ::cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cuda_matmul: cudaDeviceSynchronize failed");
    (void)err;
}

} // namespace otter
