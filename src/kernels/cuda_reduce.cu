// Must be first — provides raw_const<T> / raw_mutable<T> template bodies.
#include "dispatcher.h"
#include "cuda_kernel_engine.h"

#include "cuda_index_utils.h"
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

    // Decompose flat index into per-dimension coordinates for the input tensor.
    // Max ndim supported: 8 (matches practical tensor rank limit in this library).
    std::size_t coords[8];
    flat_to_coords(flat, in_shape, in_ndim, coords);

    // Compute offset into the input using its actual strides (may include stride-0).
    const std::size_t in_off = coords_to_offset(coords, in_strides, in_ndim);

    // Map coords to the output tensor: drop prepended dims, clamp size-1 dims to 0.
    std::size_t out_off = 0;
    for (std::size_t d = prepended; d < in_ndim; ++d) {
        const std::size_t od        = d - prepended;
        const std::size_t out_coord = (out_shape[od] == 1) ? 0 : coords[d];
        out_off += out_coord * out_strides[od];
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
    const int         block = 256;
    const int         grid  = static_cast<int>(
        (n + static_cast<std::size_t>(block) - 1) / static_cast<std::size_t>(block));
    reduce_sum_kernel<<<grid, block>>>(pa, po, n);
    cudaError_t err = ::cudaDeviceSynchronize();
    assert(err == cudaSuccess && "cuda_sum: cudaDeviceSynchronize failed");
    (void)err;
}

void CUDAKernelEngine::cuda_reduce_to(const Tensor& src, Tensor& dst) const {
    assert(dst.is_contiguous() && "cuda_reduce_to: dst must be contiguous");

    const auto& out_shape = dst.shape();
    const auto& in_shape  = src.shape();
    const std::size_t out_ndim   = out_shape.size();
    const std::size_t in_ndim    = in_shape.size();
    assert(in_ndim >= out_ndim &&
           "cuda_reduce_to: src must have >= dims than dst");
    assert(in_ndim <= 8 && "cuda_reduce_to: ndim > 8 not supported");

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

    const int block = 256;
    const int grid  = static_cast<int>(
        (numel_in + static_cast<std::size_t>(block) - 1) / static_cast<std::size_t>(block));
    reduce_to_kernel<<<grid, block>>>(sp, dp,
                                      d_in_shape, d_in_strides,
                                      d_out_shape, d_out_strides,
                                      in_ndim, prepended, numel_in);

    e = ::cudaDeviceSynchronize();
    assert(e == cudaSuccess && "cuda_reduce_to: cudaDeviceSynchronize failed"); (void)e;

    ::cudaFree(d_in_shape);
    ::cudaFree(d_in_strides);
    ::cudaFree(d_out_shape);
    ::cudaFree(d_out_strides);
}

} // namespace otter
