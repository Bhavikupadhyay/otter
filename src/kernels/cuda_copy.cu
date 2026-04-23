// Must be first — provides raw_const<T> / raw_mutable<T> template bodies.
#include "dispatcher.h"
#include "cuda_kernel_engine.h"

#include "cuda_index_utils.h"
#include "otter/tensor.h"

#include <cassert>
#include <cstddef>

#include <cuda_runtime.h>

namespace {

// Coordinate-based copy kernel: handles arbitrary src/dst strides.
// src and dst share the same logical shape. Each thread handles one logical element.
// Strides and shape are passed as device-side arrays (cudaMalloc'd by the caller).
__global__ void copy_kernel(const double* __restrict__ src,
                             double*       __restrict__ dst,
                             const std::size_t* src_strides,
                             const std::size_t* dst_strides,
                             const std::size_t* shape,
                             std::size_t ndim,
                             std::size_t numel) {
    std::size_t flat = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= numel) return;

    std::size_t in_off, out_off;
    flat_to_two_offsets(flat, shape, src_strides, dst_strides, ndim, in_off, out_off);
    dst[out_off] = src[in_off];
}

} // namespace

namespace otter {

void CUDAKernelEngine::cuda_copy(const Tensor& src, Tensor& dst) const {
    assert(src.shape() == dst.shape());
    const std::size_t ndim  = src.shape().size();
    const std::size_t numel = src.numel();

    // Copy strides and shape to device-side arrays so the kernel can access them.
    std::size_t* d_src_stride = nullptr;
    std::size_t* d_dst_stride = nullptr;
    std::size_t* d_shape      = nullptr;
    const std::size_t sz = ndim * sizeof(std::size_t);

    cudaError_t e;
    e = ::cudaMalloc(&d_src_stride, sz);
    assert(e == cudaSuccess && "cuda_copy: cudaMalloc src_stride failed"); (void)e;
    e = ::cudaMalloc(&d_dst_stride, sz);
    assert(e == cudaSuccess && "cuda_copy: cudaMalloc dst_stride failed"); (void)e;
    e = ::cudaMalloc(&d_shape, sz);
    assert(e == cudaSuccess && "cuda_copy: cudaMalloc shape failed"); (void)e;

    e = ::cudaMemcpy(d_src_stride, src.stride().data(), sz, cudaMemcpyHostToDevice);
    assert(e == cudaSuccess && "cuda_copy: cudaMemcpy src_stride failed"); (void)e;
    e = ::cudaMemcpy(d_dst_stride, dst.stride().data(), sz, cudaMemcpyHostToDevice);
    assert(e == cudaSuccess && "cuda_copy: cudaMemcpy dst_stride failed"); (void)e;
    e = ::cudaMemcpy(d_shape, src.shape().data(), sz, cudaMemcpyHostToDevice);
    assert(e == cudaSuccess && "cuda_copy: cudaMemcpy shape failed"); (void)e;

    const double* sp = raw_const<double>(src.buffer())             + src.offset();
    double*       dp = raw_mutable<double>(dst.mutable_buffer())   + dst.offset();

    const int block = 256;
    const int grid  = static_cast<int>(
        (numel + static_cast<std::size_t>(block) - 1) / static_cast<std::size_t>(block));
    copy_kernel<<<grid, block>>>(sp, dp, d_src_stride, d_dst_stride, d_shape, ndim, numel);

    e = ::cudaDeviceSynchronize();
    assert(e == cudaSuccess && "cuda_copy: cudaDeviceSynchronize failed"); (void)e;

    ::cudaFree(d_src_stride);
    ::cudaFree(d_dst_stride);
    ::cudaFree(d_shape);
}

} // namespace otter
