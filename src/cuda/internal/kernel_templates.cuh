#pragma once

// All __global__ binary/unary kernel templates and their CUDAElementwise*Kernel
// dispatcher classes. Include ONLY from cuda_kernel_engine.cu — NVCC requires
// __global__ template definitions to be in the same TU as their <<<...>>> launch
// sites.
//
// dispatcher.h must be included before this file to ensure the raw_const<T> /
// raw_mutable<T> template bodies are defined before instantiation. The including
// TU (cuda_kernel_engine.cu) guarantees this by listing dispatcher.h first.

#include "core/dispatcher.h"                 // raw_const / raw_mutable template bodies
#include "cuda/engine/cuda_kernel_engine.h"  // CUDAKernelEngine, LaunchSpec  (pragma once — safe)
#include "cuda/internal/index_utils.h"       // flat_to_offset, flat_to_two_offsets
#include "cuda/internal/functors.h"          // AddFunctor, SubFunctor, ...
#include "otter/tensor.h"        // Tensor

#include <cassert>
#include <cstddef>

#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────────────────────
// Binary __global__ kernels (external linkage required by NVCC for templates)
// ─────────────────────────────────────────────────────────────────────────────

template<typename F>
__global__ void binary_contiguous_kernel(const double* __restrict__ pa,
                                          const double* __restrict__ pb,
                                          double*       __restrict__ po,
                                          std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) po[i] = F::apply(pa[i], pb[i]);
}

// pa/pb are pre-adjusted (+ offset). shape/strides_a/strides_b are on device.
// stride-0 dims (broadcast) are handled automatically: coord * 0 == 0.
// po[flat] is valid because out is always freshly zeroed and contiguous.
template<typename F>
__global__ void binary_strided_kernel(const double*      __restrict__ pa,
                                       const double*      __restrict__ pb,
                                       double*            __restrict__ po,
                                       const std::size_t* __restrict__ shape,
                                       const std::size_t* __restrict__ strides_a,
                                       const std::size_t* __restrict__ strides_b,
                                       std::size_t ndim,
                                       std::size_t n) {
    const std::size_t flat = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= n) return;
    std::size_t off_a, off_b;
    flat_to_two_offsets(flat, shape, strides_a, strides_b, ndim, off_a, off_b);
    po[flat] = F::apply(pa[off_a], pb[off_b]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Unary __global__ kernels
// ─────────────────────────────────────────────────────────────────────────────

template<typename F>
__global__ void unary_contiguous_kernel(const double* __restrict__ pa,
                                         double*       __restrict__ po,
                                         std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) po[i] = F::apply(pa[i]);
}

// pa is pre-adjusted (+ offset). shape/strides_a are on device.
// po[flat] is valid because out is always freshly zeroed and contiguous.
template<typename F>
__global__ void unary_strided_kernel(const double*      __restrict__ pa,
                                      double*            __restrict__ po,
                                      const std::size_t* __restrict__ shape,
                                      const std::size_t* __restrict__ strides_a,
                                      std::size_t ndim,
                                      std::size_t n) {
    const std::size_t flat = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= n) return;
    const std::size_t off = flat_to_offset(flat, shape, strides_a, ndim);
    po[flat] = F::apply(pa[off]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Launch helpers and dispatcher template classes (otter namespace)
// ─────────────────────────────────────────────────────────────────────────────

namespace otter {

// ── Binary launch helpers ─────────────────────────────────────────────────────

template<typename F>
void launch_binary_contiguous(CUDAKernelEngine* engine,
                               const Tensor& a, const Tensor& b, Tensor& out,
                               const LaunchSpec& spec) {
    const double* pa = engine->raw_ptr<double>(a.buffer())               + a.offset();
    const double* pb = engine->raw_ptr<double>(b.buffer())               + b.offset();
    double*       po = engine->mutable_ptr<double>(out.mutable_buffer()) + out.offset();
    const std::size_t n     = out.numel();
    const int         block = static_cast<int>(spec.block_size);
    const int         grid  = static_cast<int>(
        (n + static_cast<std::size_t>(block) - 1) / static_cast<std::size_t>(block));
    binary_contiguous_kernel<F><<<grid, block, 0, spec.stream>>>(pa, pb, po, n);
    if (spec.sync_after) {
        const cudaError_t err = ::cudaStreamSynchronize(spec.stream);
        assert(err == cudaSuccess && "launch_binary_contiguous: cudaStreamSynchronize failed");
        (void)err;
    }
}

// a and b have been broadcast to out.shape() by BroadcastOp — their strides
// may contain zeros for broadcast dims. out is freshly allocated, contiguous.
template<typename F>
void launch_binary_strided(CUDAKernelEngine* engine,
                            const Tensor& a, const Tensor& b, Tensor& out,
                            const LaunchSpec& spec) {
    const std::size_t ndim = out.shape().size();
    const std::size_t n    = out.numel();

    std::size_t* d_shape     = nullptr;
    std::size_t* d_strides_a = nullptr;
    std::size_t* d_strides_b = nullptr;

    // cudaMallocAsync + cudaFreeAsync: stream-ordered pairing. All three are
    // queued on spec.stream so the free executes after the kernel on the same
    // stream. Requires CUDA 11.2+ (same requirement as cudaFreeAsync already
    // present in this file). Cleanup on partial failure is sequential.
    cudaError_t e;
    e = ::cudaMallocAsync(&d_shape,     ndim * sizeof(std::size_t), spec.stream);
    if (e != cudaSuccess) throw std::runtime_error(
        std::string("launch_binary_strided: cudaMallocAsync(d_shape) failed: ")
        + ::cudaGetErrorString(e));
    e = ::cudaMallocAsync(&d_strides_a, ndim * sizeof(std::size_t), spec.stream);
    if (e != cudaSuccess) {
        ::cudaFreeAsync(d_shape, spec.stream);
        throw std::runtime_error(
            std::string("launch_binary_strided: cudaMallocAsync(d_strides_a) failed: ")
            + ::cudaGetErrorString(e));
    }
    e = ::cudaMallocAsync(&d_strides_b, ndim * sizeof(std::size_t), spec.stream);
    if (e != cudaSuccess) {
        ::cudaFreeAsync(d_shape,     spec.stream);
        ::cudaFreeAsync(d_strides_a, spec.stream);
        throw std::runtime_error(
            std::string("launch_binary_strided: cudaMallocAsync(d_strides_b) failed: ")
            + ::cudaGetErrorString(e));
    }

    // Async copies queued on spec.stream — complete before the kernel reads them.
    e = ::cudaMemcpyAsync(d_shape,     out.shape().data(),  ndim * sizeof(std::size_t), cudaMemcpyHostToDevice, spec.stream); assert(e == cudaSuccess); (void)e;
    e = ::cudaMemcpyAsync(d_strides_a, a.stride().data(),   ndim * sizeof(std::size_t), cudaMemcpyHostToDevice, spec.stream); assert(e == cudaSuccess); (void)e;
    e = ::cudaMemcpyAsync(d_strides_b, b.stride().data(),   ndim * sizeof(std::size_t), cudaMemcpyHostToDevice, spec.stream); assert(e == cudaSuccess); (void)e;

    const double* pa = engine->raw_ptr<double>(a.buffer())               + a.offset();
    const double* pb = engine->raw_ptr<double>(b.buffer())               + b.offset();
    double*       po = engine->mutable_ptr<double>(out.mutable_buffer()) + out.offset();

    const int block = static_cast<int>(spec.block_size);
    const int grid  = static_cast<int>(
        (n + static_cast<std::size_t>(block) - 1) / static_cast<std::size_t>(block));
    binary_strided_kernel<F><<<grid, block, 0, spec.stream>>>(
        pa, pb, po, d_shape, d_strides_a, d_strides_b, ndim, n);

    if (spec.sync_after) {
        const cudaError_t err = ::cudaStreamSynchronize(spec.stream);
        assert(err == cudaSuccess && "launch_binary_strided: cudaStreamSynchronize failed");
        (void)err;
    }

    ::cudaFreeAsync(d_shape,     spec.stream);
    ::cudaFreeAsync(d_strides_a, spec.stream);
    ::cudaFreeAsync(d_strides_b, spec.stream);
}

// ── Unary launch helpers ──────────────────────────────────────────────────────

template<typename F>
void launch_unary_contiguous(CUDAKernelEngine* engine,
                              const Tensor& a, Tensor& out,
                              const LaunchSpec& spec) {
    const double* pa = engine->raw_ptr<double>(a.buffer())               + a.offset();
    double*       po = engine->mutable_ptr<double>(out.mutable_buffer()) + out.offset();
    const std::size_t n     = out.numel();
    const int         block = static_cast<int>(spec.block_size);
    const int         grid  = static_cast<int>(
        (n + static_cast<std::size_t>(block) - 1) / static_cast<std::size_t>(block));
    unary_contiguous_kernel<F><<<grid, block, 0, spec.stream>>>(pa, po, n);
    if (spec.sync_after) {
        const cudaError_t err = ::cudaStreamSynchronize(spec.stream);
        assert(err == cudaSuccess && "launch_unary_contiguous: cudaStreamSynchronize failed");
        (void)err;
    }
}

// a may have any strides (e.g. transposed view). out is freshly allocated, contiguous.
template<typename F>
void launch_unary_strided(CUDAKernelEngine* engine,
                           const Tensor& a, Tensor& out,
                           const LaunchSpec& spec) {
    const std::size_t ndim = out.shape().size();
    const std::size_t n    = out.numel();

    std::size_t* d_shape     = nullptr;
    std::size_t* d_strides_a = nullptr;

    cudaError_t e;
    e = ::cudaMallocAsync(&d_shape,     ndim * sizeof(std::size_t), spec.stream);
    if (e != cudaSuccess) throw std::runtime_error(
        std::string("launch_unary_strided: cudaMallocAsync(d_shape) failed: ")
        + ::cudaGetErrorString(e));
    e = ::cudaMallocAsync(&d_strides_a, ndim * sizeof(std::size_t), spec.stream);
    if (e != cudaSuccess) {
        ::cudaFreeAsync(d_shape, spec.stream);
        throw std::runtime_error(
            std::string("launch_unary_strided: cudaMallocAsync(d_strides_a) failed: ")
            + ::cudaGetErrorString(e));
    }

    e = ::cudaMemcpyAsync(d_shape,     out.shape().data(), ndim * sizeof(std::size_t), cudaMemcpyHostToDevice, spec.stream); assert(e == cudaSuccess); (void)e;
    e = ::cudaMemcpyAsync(d_strides_a, a.stride().data(),  ndim * sizeof(std::size_t), cudaMemcpyHostToDevice, spec.stream); assert(e == cudaSuccess); (void)e;

    const double* pa = engine->raw_ptr<double>(a.buffer())               + a.offset();
    double*       po = engine->mutable_ptr<double>(out.mutable_buffer()) + out.offset();

    const int block = static_cast<int>(spec.block_size);
    const int grid  = static_cast<int>(
        (n + static_cast<std::size_t>(block) - 1) / static_cast<std::size_t>(block));
    unary_strided_kernel<F><<<grid, block, 0, spec.stream>>>(
        pa, po, d_shape, d_strides_a, ndim, n);

    if (spec.sync_after) {
        const cudaError_t err = ::cudaStreamSynchronize(spec.stream);
        assert(err == cudaSuccess && "launch_unary_strided: cudaStreamSynchronize failed");
        (void)err;
    }

    ::cudaFreeAsync(d_shape,     spec.stream);
    ::cudaFreeAsync(d_strides_a, spec.stream);
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatcher template classes
// ─────────────────────────────────────────────────────────────────────────────

template<typename F>
struct CUDAElementwiseBinaryKernel final : KernelEngine::BinaryDispatcher {
    CUDAKernelEngine* engine_;

    explicit CUDAElementwiseBinaryKernel(CUDAKernelEngine* e) : engine_(e) {}

    void call(const Tensor& a, const Tensor& b, Tensor& out) const override {
        // Read default_spec_ at call time so stream/sync_after changes in the
        // engine (e.g. after CUDABackend sets the stream) are picked up correctly.
        const LaunchSpec& spec = engine_->default_spec_;
        if (a.is_contiguous() && b.is_contiguous()) {
            launch_binary_contiguous<F>(engine_, a, b, out, spec);
        } else {
            launch_binary_strided<F>(engine_, a, b, out, spec);
        }
    }
};

template<typename F>
struct CUDAElementwiseUnaryKernel final : KernelEngine::UnaryDispatcher {
    CUDAKernelEngine* engine_;

    explicit CUDAElementwiseUnaryKernel(CUDAKernelEngine* e) : engine_(e) {}

    void call(const Tensor& a, Tensor& out) const override {
        const LaunchSpec& spec = engine_->default_spec_;
        if (a.is_contiguous()) {
            launch_unary_contiguous<F>(engine_, a, out, spec);
        } else {
            launch_unary_strided<F>(engine_, a, out, spec);
        }
    }
};

} // namespace otter
