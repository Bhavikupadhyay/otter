// Must be first — provides raw_const<T> / raw_mutable<T> template bodies.
#include "dispatcher.h"
#include "cuda_kernel_engine.h"

#include "functors.h"
#include "otter/tensor.h"

#include <cassert>
#include <cstddef>

#include <cuda_runtime.h>

namespace {

// ── Binary kernel — all three tensors must be contiguous (caller asserts) ─────
//
// Calls F::apply(a[i], b[i]) — no functor instance passed through register args.

template<typename F>
__global__ void binary_kernel(const double* __restrict__ a,
                               const double* __restrict__ b,
                               double*       __restrict__ out,
                               std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = F::apply(a[i], b[i]);
}

} // namespace

namespace otter {

namespace {

template<typename F>
void launch_binary(const double* pa, const double* pb, double* po,
                   std::size_t n, const LaunchSpec& spec) {
    const int block = static_cast<int>(spec.block_size);
    const int grid  = static_cast<int>(
        (n + static_cast<std::size_t>(block) - 1) / static_cast<std::size_t>(block));
    binary_kernel<F><<<grid, block, 0, spec.stream>>>(pa, pb, po, n);
    if (spec.sync_after) {
        const cudaError_t err = ::cudaDeviceSynchronize();
        assert(err == cudaSuccess && "cuda_binary: cudaDeviceSynchronize failed");
        (void)err;
    }
}

} // namespace

// ── Public cuda_* methods ─────────────────────────────────────────────────────

void CUDAKernelEngine::cuda_add(const Tensor& a, const Tensor& b, Tensor& out) const {
    assert(a.is_contiguous() && b.is_contiguous() && out.is_contiguous());
    launch_binary<AddFunctor>(raw_const<double>(a.buffer())             + a.offset(),
                              raw_const<double>(b.buffer())             + b.offset(),
                              raw_mutable<double>(out.mutable_buffer()) + out.offset(),
                              out.numel(), default_spec_);
}

void CUDAKernelEngine::cuda_sub(const Tensor& a, const Tensor& b, Tensor& out) const {
    assert(a.is_contiguous() && b.is_contiguous() && out.is_contiguous());
    launch_binary<SubFunctor>(raw_const<double>(a.buffer())             + a.offset(),
                              raw_const<double>(b.buffer())             + b.offset(),
                              raw_mutable<double>(out.mutable_buffer()) + out.offset(),
                              out.numel(), default_spec_);
}

void CUDAKernelEngine::cuda_mul(const Tensor& a, const Tensor& b, Tensor& out) const {
    assert(a.is_contiguous() && b.is_contiguous() && out.is_contiguous());
    launch_binary<MulFunctor>(raw_const<double>(a.buffer())             + a.offset(),
                              raw_const<double>(b.buffer())             + b.offset(),
                              raw_mutable<double>(out.mutable_buffer()) + out.offset(),
                              out.numel(), default_spec_);
}

void CUDAKernelEngine::cuda_div(const Tensor& a, const Tensor& b, Tensor& out) const {
    assert(a.is_contiguous() && b.is_contiguous() && out.is_contiguous());
    launch_binary<DivFunctor>(raw_const<double>(a.buffer())             + a.offset(),
                              raw_const<double>(b.buffer())             + b.offset(),
                              raw_mutable<double>(out.mutable_buffer()) + out.offset(),
                              out.numel(), default_spec_);
}

} // namespace otter
