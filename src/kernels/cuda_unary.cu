// Must be first — provides raw_const<T> / raw_mutable<T> template bodies.
#include "dispatcher.h"
#include "cuda_kernel_engine.h"

#include "otter/tensor.h"

#include <cassert>
#include <cstddef>
#include <cmath>

#include <cuda_runtime.h>

namespace {

// ── Functor structs — __device__ only ─────────────────────────────────────────

struct NegOp      { __device__ double operator()(double x) const noexcept { return -x; } };
struct ExpOp      { __device__ double operator()(double x) const noexcept { return ::exp(x); } };
struct LogOp      { __device__ double operator()(double x) const noexcept { return ::log(x); } };
struct SqrtOp     { __device__ double operator()(double x) const noexcept { return ::sqrt(x); } };
struct ReluOp     { __device__ double operator()(double x) const noexcept { return x > 0.0 ? x : 0.0; } };
struct ReluMaskOp { __device__ double operator()(double x) const noexcept { return x > 0.0 ? 1.0 : 0.0; } };

// ── Unary kernel — both tensors must be contiguous (caller asserts) ───────────

template<typename Op>
__global__ void unary_kernel(const double* __restrict__ a,
                              double*       __restrict__ out,
                              std::size_t n, Op op) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = op(a[i]);
}

} // namespace

namespace otter {

namespace {

template<typename Op>
void launch_unary(const double* pa, double* po, std::size_t n, Op op,
                  const LaunchSpec& spec) {
    const int block = static_cast<int>(spec.block_size);
    const int grid  = static_cast<int>(
        (n + static_cast<std::size_t>(block) - 1) / static_cast<std::size_t>(block));
    unary_kernel<<<grid, block, 0, spec.stream>>>(pa, po, n, op);
    if (spec.sync_after) {
        cudaError_t err = ::cudaDeviceSynchronize();
        assert(err == cudaSuccess && "cuda_unary: cudaDeviceSynchronize failed");
        (void)err;
    }
}

} // namespace

void CUDAKernelEngine::cuda_neg(const Tensor& a, Tensor& out) const {
    assert(a.is_contiguous() && out.is_contiguous());
    launch_unary(raw_const<double>(a.buffer())             + a.offset(),
                 raw_mutable<double>(out.mutable_buffer()) + out.offset(),
                 a.numel(), NegOp{}, default_spec_);
}

void CUDAKernelEngine::cuda_exp(const Tensor& a, Tensor& out) const {
    assert(a.is_contiguous() && out.is_contiguous());
    launch_unary(raw_const<double>(a.buffer())             + a.offset(),
                 raw_mutable<double>(out.mutable_buffer()) + out.offset(),
                 a.numel(), ExpOp{}, default_spec_);
}

void CUDAKernelEngine::cuda_log(const Tensor& a, Tensor& out) const {
    // IEEE 754: x=0 → -inf, x<0 → nan — no check needed.
    assert(a.is_contiguous() && out.is_contiguous());
    launch_unary(raw_const<double>(a.buffer())             + a.offset(),
                 raw_mutable<double>(out.mutable_buffer()) + out.offset(),
                 a.numel(), LogOp{}, default_spec_);
}

void CUDAKernelEngine::cuda_sqrt(const Tensor& a, Tensor& out) const {
    // IEEE 754: x<0 → nan — no check needed.
    assert(a.is_contiguous() && out.is_contiguous());
    launch_unary(raw_const<double>(a.buffer())             + a.offset(),
                 raw_mutable<double>(out.mutable_buffer()) + out.offset(),
                 a.numel(), SqrtOp{}, default_spec_);
}

void CUDAKernelEngine::cuda_relu(const Tensor& a, Tensor& out) const {
    // At x=0: output is 0.0 (right-hand derivative convention, matches PyTorch).
    assert(a.is_contiguous() && out.is_contiguous());
    launch_unary(raw_const<double>(a.buffer())             + a.offset(),
                 raw_mutable<double>(out.mutable_buffer()) + out.offset(),
                 a.numel(), ReluOp{}, default_spec_);
}

void CUDAKernelEngine::cuda_relu_mask(const Tensor& a, Tensor& out) const {
    // 1.0 where input > 0, 0.0 otherwise. At x=0: mask = 0.0 (matches PyTorch).
    assert(a.is_contiguous() && out.is_contiguous());
    launch_unary(raw_const<double>(a.buffer())             + a.offset(),
                 raw_mutable<double>(out.mutable_buffer()) + out.offset(),
                 a.numel(), ReluMaskOp{}, default_spec_);
}

} // namespace otter
