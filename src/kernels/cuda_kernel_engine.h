#pragma once

// Internal header — not part of the public include/otter/ surface.
// Only included by src/kernels/ translation units and cuda_backend.cpp.

#include "otter/kernel/kernel_engine.h"

#include <cuda_runtime.h>

namespace otter {

class Tensor;
class Buffer;

// Launch parameters for CUDA kernels.
// Defaults reproduce the original hardcoded behaviour (block=256, no stream,
// sync after every launch). Override per-engine or per-call as needed.
struct LaunchSpec {
    std::size_t  block_size = 256;
    cudaStream_t stream     = nullptr;
    bool         sync_after = true;
};

// CUDAKernelEngine — concrete KernelEngine for the CUDA backend.
//
// Registers one dispatcher per kernel; each calls back into a cuda_* method
// below. cuda_* methods are member functions so they can reach the protected
// raw_const<T> / raw_mutable<T> helpers from KernelEngine.
//
// Include dispatcher.h (not this header) when writing .cu files that call
// raw_const / raw_mutable — dispatcher.h supplies the template bodies.
class CUDAKernelEngine final : public KernelEngine {
public:
    CUDAKernelEngine();  // registers dispatchers; expands one kernel per sprint

    LaunchSpec default_spec_{};  // configurable; defaults match prior hardcoded values

    // ── Kernel implementations — called by registered CUDA dispatchers only ───
    void   cuda_fill          (Tensor& t, double value) const;
    double cuda_element_read  (const Tensor& t, std::size_t flat_idx) const;
    void   cuda_bulk_host_read(const Tensor& src, std::vector<double>& dst) const;

    // Binary element-wise
    void cuda_add(const Tensor& a, const Tensor& b, Tensor& out) const;
    void cuda_sub(const Tensor& a, const Tensor& b, Tensor& out) const;
    void cuda_mul(const Tensor& a, const Tensor& b, Tensor& out) const;
    void cuda_div(const Tensor& a, const Tensor& b, Tensor& out) const;

    // Unary element-wise
    void cuda_neg      (const Tensor& a, Tensor& out) const;
    void cuda_exp      (const Tensor& a, Tensor& out) const;
    void cuda_log      (const Tensor& a, Tensor& out) const;
    void cuda_sqrt     (const Tensor& a, Tensor& out) const;
    void cuda_relu     (const Tensor& a, Tensor& out) const;
    void cuda_relu_mask(const Tensor& a, Tensor& out) const;

    // In-place
    void cuda_scale(Tensor& dst, double alpha) const;
    void cuda_axpy (Tensor& dst, double alpha, const Tensor& src) const;

    // Copy (non-contiguous src → contiguous dst)
    void cuda_copy(const Tensor& src, Tensor& dst) const;

    // Reductions
    void cuda_sum      (const Tensor& a, Tensor& out) const;
    void cuda_reduce_to(const Tensor& src, Tensor& dst) const;

    // Matrix multiply
    void cuda_matmul(const Tensor& a, const Tensor& b, Tensor& out) const;
};

} // namespace otter
