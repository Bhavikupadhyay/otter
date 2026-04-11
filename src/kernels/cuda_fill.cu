// Must be first — provides raw_const<T> / raw_mutable<T> template bodies.
#include "dispatcher.h"
#include "cuda_kernel_engine.h"

#include "otter/tensor.h"

#include <cassert>
#include <cstddef>

#include <cuda_runtime.h>

namespace {

__global__ void fill_kernel(double* __restrict__ data, double value, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) data[i] = value;
}

} // namespace

namespace otter {

void CUDAKernelEngine::cuda_fill(Tensor& t, double value) const {
    double*           ptr   = raw_mutable<double>(t.mutable_buffer());
    const std::size_t n     = t.numel();
    const int         block = 256;
    const int         grid  = static_cast<int>((n + static_cast<std::size_t>(block) - 1)
                                               / static_cast<std::size_t>(block));
    fill_kernel<<<grid, block>>>(ptr + t.offset(), value, n);

    // Synchronize so the fill is complete before returning. Callers (e.g. fill_())
    // may immediately read the tensor on the host via at() / to_vector().
    cudaError_t err = ::cudaDeviceSynchronize();
    assert(err == cudaSuccess && "CUDAKernelEngine::cuda_fill: cudaDeviceSynchronize failed");
    (void)err;
}

double CUDAKernelEngine::cuda_element_read(const Tensor& t, std::size_t flat_idx) const {
    // flat_idx is the pre-computed buffer index from Tensor::at() (offset + stride walk).
    // Sync ensures any preceding device-side writes are visible to the host.
    cudaError_t err = ::cudaDeviceSynchronize();
    assert(err == cudaSuccess &&
           "CUDAKernelEngine::cuda_element_read: cudaDeviceSynchronize failed");
    (void)err;
    return raw_const<double>(t.buffer())[flat_idx];
}

void CUDAKernelEngine::cuda_bulk_host_read(const Tensor& src,
                                            std::vector<double>& dst) const {
    // Precondition: src is contiguous — caller (Tensor::to) calls contiguous() first.
    assert(src.is_contiguous());
    // One sync for the whole read; unified memory is host-accessible after this.
    cudaError_t err = ::cudaDeviceSynchronize();
    assert(err == cudaSuccess &&
           "CUDAKernelEngine::cuda_bulk_host_read: cudaDeviceSynchronize failed");
    (void)err;
    const double* ptr = raw_const<double>(src.buffer()) + src.offset();
    dst.assign(ptr, ptr + src.numel());
}

} // namespace otter
