// Must be first — provides raw_const<T> / raw_mutable<T> template bodies.
#include "dispatcher.h"
#include "cuda_kernel_engine.h"

#include "otter/tensor.h"

#include <cassert>
#include <cstddef>

#include <cuda_runtime.h>

namespace {

__global__ void scale_kernel(double* __restrict__ dst, double alpha, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] *= alpha;
}

// dst[i] += alpha * src[i]
__global__ void axpy_kernel(double* __restrict__ dst, double alpha,
                             const double* __restrict__ src, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += alpha * src[i];
}

} // namespace

namespace otter {

void CUDAKernelEngine::cuda_scale(Tensor& dst, double alpha) const {
    assert(dst.is_contiguous());
    const std::size_t n     = dst.numel();
    double*           ptr   = raw_mutable<double>(dst.mutable_buffer()) + dst.offset();
    const int         block = 256;
    const int         grid  = static_cast<int>(
        (n + static_cast<std::size_t>(block) - 1) / static_cast<std::size_t>(block));
    scale_kernel<<<grid, block>>>(ptr, alpha, n);
    cudaError_t err = ::cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cuda_scale: cudaDeviceSynchronize failed");
    (void)err;
}

void CUDAKernelEngine::cuda_axpy(Tensor& dst, double alpha, const Tensor& src) const {
    assert(dst.is_contiguous() && src.is_contiguous());
    assert(dst.shape() == src.shape());
    const std::size_t n    = dst.numel();
    double*           dp   = raw_mutable<double>(dst.mutable_buffer()) + dst.offset();
    const double*     sp   = raw_const<double>(src.buffer())           + src.offset();
    const int         block = 256;
    const int         grid  = static_cast<int>(
        (n + static_cast<std::size_t>(block) - 1) / static_cast<std::size_t>(block));
    axpy_kernel<<<grid, block>>>(dp, alpha, sp, n);
    cudaError_t err = ::cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cuda_axpy: cudaDeviceSynchronize failed");
    (void)err;
}

} // namespace otter
