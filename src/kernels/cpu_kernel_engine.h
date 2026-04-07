#pragma once

// Internal header — not part of the public include/otter/ surface.
// Only included by src/kernels/cpu_kernel_engine.cpp and src/backends/cpu_backend.cpp.

#include "otter/kernel/kernel_engine.h"

namespace otter {

class Tensor;
class Buffer;

// CPUKernelEngine — concrete KernelEngine for the CPU backend.
//
// Registers dispatchers for all basic ops in its constructor.
// Each dispatcher stores a CPUKernelEngine* and forwards calls to the
// cpu_* methods below. Those methods are KernelEngine subclass methods,
// so they can call the protected raw_const<T> / raw_mutable<T> helpers.
//
// Include dispatcher.h (not this header) when writing .cpp files that
// call raw_const / raw_mutable — dispatcher.h supplies the template bodies.
class CPUKernelEngine final : public KernelEngine {
public:
    CPUKernelEngine();  // registers all dispatchers

    // ── Kernel implementations ── called by registered CPU dispatchers only ──
    //
    // Protected raw_const / raw_mutable are accessible here because
    // CPUKernelEngine is a KernelEngine subclass.

    void   cpu_fill        (Tensor& t, double value) const;
    void   cpu_add         (const Tensor& a, const Tensor& b, Tensor& out) const;
    void   cpu_mul         (const Tensor& a, const Tensor& b, Tensor& out) const;
    void   cpu_neg         (const Tensor& a, Tensor& out) const;
    void   cpu_sum         (const Tensor& a, Tensor& out) const;
    void   cpu_copy        (const Tensor& src, Tensor& dst) const;
    void   cpu_reduce_to   (const Tensor& src, Tensor& dst) const;
    void   cpu_matmul      (const Tensor& a, const Tensor& b, Tensor& out) const;
    double cpu_element_read(const Tensor& t, std::size_t flat_idx) const;
};

} // namespace otter
