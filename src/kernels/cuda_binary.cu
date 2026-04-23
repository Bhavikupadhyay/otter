// Must be first — provides raw_const<T> / raw_mutable<T> template bodies.
#include "dispatcher.h"
#include "cuda_kernel_engine.h"

#include "otter/tensor.h"

#include <cassert>
#include <cstddef>

#include <cuda_runtime.h>

namespace {

// ── Functor structs — __device__ only, no lambdas (no --expt-extended-lambda) ──

struct AddOp { __device__ double operator()(double a, double b) const noexcept { return a + b; } };
struct SubOp { __device__ double operator()(double a, double b) const noexcept { return a - b; } };
struct MulOp { __device__ double operator()(double a, double b) const noexcept { return a * b; } };
struct DivOp { __device__ double operator()(double a, double b) const noexcept { return a / b; } };

// ── Binary kernel — all three tensors must be contiguous (caller asserts) ─────

template<typename Op>
__global__ void binary_kernel(const double* __restrict__ a,
                               const double* __restrict__ b,
                               double*       __restrict__ out,
                               std::size_t n, Op op) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = op(a[i], b[i]);
}

} // namespace

namespace otter {

// ── Shared launch helper (internal linkage) ───────────────────────────────────

namespace {

template<typename Op>
void launch_binary(const double* pa, const double* pb, double* po,
                   std::size_t n, Op op, const LaunchSpec& spec) {
    const int block = static_cast<int>(spec.block_size);
    const int grid  = static_cast<int>(
        (n + static_cast<std::size_t>(block) - 1) / static_cast<std::size_t>(block));
    binary_kernel<<<grid, block, 0, spec.stream>>>(pa, pb, po, n, op);
    if (spec.sync_after) {
        cudaError_t err = ::cudaDeviceSynchronize();
        assert(err == cudaSuccess && "cuda_binary: cudaDeviceSynchronize failed");
        (void)err;
    }
}

} // namespace

// ── Public cuda_* methods ─────────────────────────────────────────────────────

void CUDAKernelEngine::cuda_add(const Tensor& a, const Tensor& b, Tensor& out) const {
    assert(a.is_contiguous() && b.is_contiguous() && out.is_contiguous());
    launch_binary(raw_const<double>(a.buffer())             + a.offset(),
                  raw_const<double>(b.buffer())             + b.offset(),
                  raw_mutable<double>(out.mutable_buffer()) + out.offset(),
                  out.numel(), AddOp{}, default_spec_);
}

void CUDAKernelEngine::cuda_sub(const Tensor& a, const Tensor& b, Tensor& out) const {
    assert(a.is_contiguous() && b.is_contiguous() && out.is_contiguous());
    launch_binary(raw_const<double>(a.buffer())             + a.offset(),
                  raw_const<double>(b.buffer())             + b.offset(),
                  raw_mutable<double>(out.mutable_buffer()) + out.offset(),
                  out.numel(), SubOp{}, default_spec_);
}

void CUDAKernelEngine::cuda_mul(const Tensor& a, const Tensor& b, Tensor& out) const {
    assert(a.is_contiguous() && b.is_contiguous() && out.is_contiguous());
    launch_binary(raw_const<double>(a.buffer())             + a.offset(),
                  raw_const<double>(b.buffer())             + b.offset(),
                  raw_mutable<double>(out.mutable_buffer()) + out.offset(),
                  out.numel(), MulOp{}, default_spec_);
}

void CUDAKernelEngine::cuda_div(const Tensor& a, const Tensor& b, Tensor& out) const {
    // IEEE 754: b=0 produces ±inf or nan — no check needed.
    assert(a.is_contiguous() && b.is_contiguous() && out.is_contiguous());
    launch_binary(raw_const<double>(a.buffer())             + a.offset(),
                  raw_const<double>(b.buffer())             + b.offset(),
                  raw_mutable<double>(out.mutable_buffer()) + out.offset(),
                  out.numel(), DivOp{}, default_spec_);
}

} // namespace otter
