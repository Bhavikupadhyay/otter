// Must be first — provides raw_const<T> / raw_mutable<T> template bodies.
#include "dispatcher.h"
#include "cuda_kernel_engine.h"

#include "otter/tensor.h"
#include "otter/detail/stride_utils.h"

#include <cassert>
#include <cstddef>

#include <cuda_runtime.h>

namespace {

// ReduceSum: each thread atomicAdd its element into the single output scalar.
// Requires compute capability >= 6.0 for double atomicAdd.
// out must be pre-zeroed (caller ensures this via Tensor::zeros).
__global__ void reduce_sum_kernel(const double* __restrict__ a,
                                   double*                    out,
                                   std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(out, a[i]);
}

// ReduceTo: scatter-accumulate to target shape (broadcast backward kernel).
// dst must be pre-zeroed (caller ensures this).
// Requires compute capability >= 6.0 for double atomicAdd.
//
// in_ndim / in_shape / in_strides: the incoming gradient tensor (broadcast shape).
// prepended: number of extra leading dims in src beyond out_ndim.
// out_shape / out_strides: the original pre-broadcast shape (contiguous layout).
__global__ void reduce_to_kernel(const double*      __restrict__ src,
                                  double*                         dst,
                                  const std::size_t* in_shape,
                                  const std::size_t* in_strides,
                                  const std::size_t* out_shape,
                                  const std::size_t* out_strides,
                                  std::size_t        in_ndim,
                                  std::size_t        prepended,
                                  std::size_t        numel_in) {
    std::size_t flat = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= numel_in) return;

    // Single decomposition pass: compute in_off and out_off together.
    // in_off uses in_strides directly (handles stride-0 broadcast dims).
    // out_off skips prepended dims and clamps size-1 dims to 0.
    std::size_t in_off = 0, out_off = 0;
    std::size_t rem = flat;
    for (int d = static_cast<int>(in_ndim) - 1; d >= 0; --d) {
        const std::size_t ud    = static_cast<std::size_t>(d);
        const std::size_t coord = rem % in_shape[ud];
        rem                    /= in_shape[ud];
        in_off                 += coord * in_strides[ud];
        if (ud >= prepended) {
            const std::size_t od        = ud - prepended;
            const std::size_t out_coord = (out_shape[od] == 1) ? 0 : coord;
            out_off                    += out_coord * out_strides[od];
        }
    }

    atomicAdd(dst + out_off, src[in_off]);
}

} // namespace

namespace otter {

void CUDAKernelEngine::cuda_sum(const Tensor& a, Tensor& out) const {
    assert(a.is_contiguous() &&
           "cuda_sum: input must be contiguous; call contiguous() before dispatch");
    const double*     pa    = raw_const<double>(a.buffer())             + a.offset();
    double*           po    = raw_mutable<double>(out.mutable_buffer()) + out.offset();
    const std::size_t n     = a.numel();
    const int         block = static_cast<int>(default_spec_.block_size);
    const int         grid  = static_cast<int>(
        (n + static_cast<std::size_t>(block) - 1) / static_cast<std::size_t>(block));
    reduce_sum_kernel<<<grid, block, 0, default_spec_.stream>>>(pa, po, n);
    if (default_spec_.sync_after) {
        cudaError_t err = ::cudaDeviceSynchronize();
        assert(err == cudaSuccess && "cuda_sum: cudaDeviceSynchronize failed");
        (void)err;
    }
}

void CUDAKernelEngine::cuda_reduce_to(const Tensor& src, Tensor& dst) const {
    assert(dst.is_contiguous() && "cuda_reduce_to: dst must be contiguous");

    const auto& out_shape = dst.shape();
    const auto& in_shape  = src.shape();
    const std::size_t out_ndim   = out_shape.size();
    const std::size_t in_ndim    = in_shape.size();
    assert(in_ndim >= out_ndim &&
           "cuda_reduce_to: src must have >= dims than dst");

    const std::size_t prepended  = in_ndim - out_ndim;
    const std::size_t numel_in   = src.numel();

    // Contiguous strides for the output (dst is contiguous by assertion above).
    const auto out_strides = detail::contiguous_strides(out_shape);

    // Allocate device-side arrays for shape and stride metadata.
    std::size_t* d_in_shape    = nullptr;
    std::size_t* d_in_strides  = nullptr;
    std::size_t* d_out_shape   = nullptr;
    std::size_t* d_out_strides = nullptr;

    cudaError_t e;
    e = ::cudaMalloc(&d_in_shape,    in_ndim  * sizeof(std::size_t));
    assert(e == cudaSuccess); (void)e;
    e = ::cudaMalloc(&d_in_strides,  in_ndim  * sizeof(std::size_t));
    assert(e == cudaSuccess); (void)e;
    e = ::cudaMalloc(&d_out_shape,   out_ndim * sizeof(std::size_t));
    assert(e == cudaSuccess); (void)e;
    e = ::cudaMalloc(&d_out_strides, out_ndim * sizeof(std::size_t));
    assert(e == cudaSuccess); (void)e;

    e = ::cudaMemcpy(d_in_shape,    in_shape.data(),       in_ndim  * sizeof(std::size_t), cudaMemcpyHostToDevice);
    assert(e == cudaSuccess); (void)e;
    e = ::cudaMemcpy(d_in_strides,  src.stride().data(),   in_ndim  * sizeof(std::size_t), cudaMemcpyHostToDevice);
    assert(e == cudaSuccess); (void)e;
    e = ::cudaMemcpy(d_out_shape,   out_shape.data(),      out_ndim * sizeof(std::size_t), cudaMemcpyHostToDevice);
    assert(e == cudaSuccess); (void)e;
    e = ::cudaMemcpy(d_out_strides, out_strides.data(),    out_ndim * sizeof(std::size_t), cudaMemcpyHostToDevice);
    assert(e == cudaSuccess); (void)e;

    const double* sp = raw_const<double>(src.buffer())             + src.offset();
    double*       dp = raw_mutable<double>(dst.mutable_buffer())   + dst.offset();

    const int block = static_cast<int>(default_spec_.block_size);
    const int grid  = static_cast<int>(
        (numel_in + static_cast<std::size_t>(block) - 1) / static_cast<std::size_t>(block));
    reduce_to_kernel<<<grid, block, 0, default_spec_.stream>>>(sp, dp,
                                                                d_in_shape, d_in_strides,
                                                                d_out_shape, d_out_strides,
                                                                in_ndim, prepended, numel_in);

    if (default_spec_.sync_after) {
        e = ::cudaDeviceSynchronize();
        assert(e == cudaSuccess && "cuda_reduce_to: cudaDeviceSynchronize failed"); (void)e;
    }

    ::cudaFree(d_in_shape);
    ::cudaFree(d_in_strides);
    ::cudaFree(d_out_shape);
    ::cudaFree(d_out_strides);
}

} // namespace otter
