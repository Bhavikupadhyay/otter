#pragma once

// Internal header — not part of the public include/otter/ surface.
// Only included by src/kernels/ translation units and cuda_backend.cpp.

#include "otter/kernel/kernel_engine.h"

namespace otter {

class Tensor;
class Buffer;

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

    // ── Kernel implementations — called by registered CUDA dispatchers only ───
    void   cuda_fill          (Tensor& t, double value) const;
    double cuda_element_read  (const Tensor& t, std::size_t flat_idx) const;
    void   cuda_bulk_host_read(const Tensor& src, std::vector<double>& dst) const;
};

} // namespace otter
