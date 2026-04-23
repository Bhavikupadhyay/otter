// Must be first — provides raw_const<T> / raw_mutable<T> template bodies.
#include "dispatcher.h"
#include "cuda_kernel_engine.h"

#include "functors.h"
#include "otter/tensor.h"

#include <cassert>
#include <cstddef>

#include <cuda_runtime.h>

namespace {

// ── Unary kernel — both tensors must be contiguous (caller asserts) ───────────
//
// Calls F::apply(a[i]) — no functor instance passed through register args.

template<typename F>
__global__ void unary_kernel(const double* __restrict__ a,
                              double*       __restrict__ out,
                              std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = F::apply(a[i]);
}

} // namespace

namespace otter {

namespace {

template<typename F>
void launch_unary(const double* pa, double* po, std::size_t n, const LaunchSpec& spec) {
    const int block = static_cast<int>(spec.block_size);
    const int grid  = static_cast<int>(
        (n + static_cast<std::size_t>(block) - 1) / static_cast<std::size_t>(block));
    unary_kernel<F><<<grid, block, 0, spec.stream>>>(pa, po, n);
    if (spec.sync_after) {
        const cudaError_t err = ::cudaDeviceSynchronize();
        assert(err == cudaSuccess && "cuda_unary: cudaDeviceSynchronize failed");
        (void)err;
    }
}

} // namespace

void CUDAKernelEngine::cuda_neg(const Tensor& a, Tensor& out) const {
    assert(a.is_contiguous() && out.is_contiguous());
    launch_unary<NegFunctor>(raw_const<double>(a.buffer())             + a.offset(),
                             raw_mutable<double>(out.mutable_buffer()) + out.offset(),
                             a.numel(), default_spec_);
}

void CUDAKernelEngine::cuda_exp(const Tensor& a, Tensor& out) const {
    assert(a.is_contiguous() && out.is_contiguous());
    launch_unary<ExpFunctor>(raw_const<double>(a.buffer())             + a.offset(),
                             raw_mutable<double>(out.mutable_buffer()) + out.offset(),
                             a.numel(), default_spec_);
}

void CUDAKernelEngine::cuda_log(const Tensor& a, Tensor& out) const {
    assert(a.is_contiguous() && out.is_contiguous());
    launch_unary<LogFunctor>(raw_const<double>(a.buffer())             + a.offset(),
                             raw_mutable<double>(out.mutable_buffer()) + out.offset(),
                             a.numel(), default_spec_);
}

void CUDAKernelEngine::cuda_sqrt(const Tensor& a, Tensor& out) const {
    assert(a.is_contiguous() && out.is_contiguous());
    launch_unary<SqrtFunctor>(raw_const<double>(a.buffer())             + a.offset(),
                              raw_mutable<double>(out.mutable_buffer()) + out.offset(),
                              a.numel(), default_spec_);
}

void CUDAKernelEngine::cuda_relu(const Tensor& a, Tensor& out) const {
    assert(a.is_contiguous() && out.is_contiguous());
    launch_unary<ReluFunctor>(raw_const<double>(a.buffer())             + a.offset(),
                              raw_mutable<double>(out.mutable_buffer()) + out.offset(),
                              a.numel(), default_spec_);
}

void CUDAKernelEngine::cuda_relu_mask(const Tensor& a, Tensor& out) const {
    assert(a.is_contiguous() && out.is_contiguous());
    launch_unary<ReluMaskFunctor>(raw_const<double>(a.buffer())             + a.offset(),
                                  raw_mutable<double>(out.mutable_buffer()) + out.offset(),
                                  a.numel(), default_spec_);
}

} // namespace otter
