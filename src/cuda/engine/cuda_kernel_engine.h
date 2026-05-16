#pragma once

// Internal header — not part of the public include/otter/ surface.
// Only included by src/cuda/ translation units.

#include "otter/kernel/kernel_engine.h"
#include "cuda/engine/launch_spec.h"

namespace otter {

class Tensor;
class Buffer;

// CUDAKernelEngine — concrete KernelEngine for the CUDA backend.
//
// Binary and unary element-wise kernels are dispatched through template classes
// (CUDAElementwiseBinaryKernel<F>, CUDAElementwiseUnaryKernel<F>) registered in
// the constructor. Those classes call raw_ptr / mutable_ptr below instead of the
// protected raw_const / raw_mutable directly, which would be inaccessible from a
// non-member template class.
//
// The remaining kernel families (fill, copy, reduce, matmul, inplace) keep their
// engine-method pattern until a follow-up sprint migrates them.
//
// Include core/dispatcher.h (not this header) when writing .cu files that call
// raw_const / raw_mutable — dispatcher.h supplies the template bodies.
class CUDAKernelEngine final : public KernelEngine {
public:
    CUDAKernelEngine();  // registers all dispatchers

    // Written once during single-threaded backend construction (CUDABackend ctor
    // sets stream; CUDAKernelEngine ctor sets sync_after). Reads by kernel
    // dispatchers are safe only because no write occurs after construction.
    // Any future write from a non-construction context requires external
    // synchronization — there is no lock protecting this field.
    LaunchSpec default_spec_{};

    // ── Forwarding accessors for template dispatcher classes ──────────────────
    // Template dispatcher instances are not KernelEngine subclasses and therefore
    // cannot call the protected raw_const / raw_mutable directly. These thin
    // wrappers delegate through *this (legal as member functions).
    template<typename T>
    const T* raw_ptr(const Buffer& b) const { return raw_const<T>(b); }

    template<typename T>
    T* mutable_ptr(Buffer& b) const { return raw_mutable<T>(b); }

    // ── Kernel implementations — called by registered CUDA dispatchers only ───
    void   cuda_fill          (Tensor& t, double value) const;
    double cuda_element_read  (const Tensor& t, std::size_t flat_idx) const;
    void   cuda_bulk_host_read(const Tensor& src, std::vector<double>& dst) const;

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
