// dispatcher.h MUST be first — provides raw_const<T> / raw_mutable<T> template
// bodies. cuda_kernel_engine.h pulls in kernel_engine.h (which declares them);
// dispatcher.h bridges buffer.h to provide the definitions.
#include "core/dispatcher.h"
#include "cuda/engine/cuda_kernel_engine.h"

// cuda/internal/kernel_templates.cuh defines all __global__ binary/unary kernel
// templates and their CUDAElementwise*Kernel dispatcher classes. NVCC requires the
// template definitions and their <<<...>>> launch sites in the same TU.
#include "cuda/internal/kernel_templates.cuh"

#include "otter/tensor.h"

#include <memory>

namespace otter {

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Retained engine-method dispatcher structs
// (fill / element_read / bulk_host_read / copy / reduce / matmul / inplace)
// Binary and unary ops now use CUDAElementwiseBinaryKernel<F> /
// CUDAElementwiseUnaryKernel<F> from cuda_kernel_templates.cuh.
// ─────────────────────────────────────────────────────────────────────────────

struct CUDAFillDispatcher final : KernelEngine::FillDispatcher {
    CUDAKernelEngine* engine_;
    explicit CUDAFillDispatcher(CUDAKernelEngine* e) : engine_(e) {}
    void call(Tensor& t, double value) const override {
        engine_->cuda_fill(t, value);
    }
};

struct CUDAElementReadDispatcher final : KernelEngine::ElementReadDispatcher {
    CUDAKernelEngine* engine_;
    explicit CUDAElementReadDispatcher(CUDAKernelEngine* e) : engine_(e) {}
    double call(const Tensor& t, std::size_t flat_idx) const override {
        return engine_->cuda_element_read(t, flat_idx);
    }
};

struct CUDABulkHostReadDispatcher final : KernelEngine::BulkHostReadDispatcher {
    CUDAKernelEngine* engine_;
    explicit CUDABulkHostReadDispatcher(CUDAKernelEngine* e) : engine_(e) {}
    void call(const Tensor& src, std::vector<double>& dst) const override {
        engine_->cuda_bulk_host_read(src, dst);
    }
};

struct CUDAReduceSumDispatcher final : KernelEngine::UnaryDispatcher {
    CUDAKernelEngine* engine_;
    explicit CUDAReduceSumDispatcher(CUDAKernelEngine* e) : engine_(e) {}
    void call(const Tensor& a, Tensor& out) const override {
        engine_->cuda_sum(a, out);
    }
};

struct CUDAReduceToDispatcher final : KernelEngine::UnaryDispatcher {
    CUDAKernelEngine* engine_;
    explicit CUDAReduceToDispatcher(CUDAKernelEngine* e) : engine_(e) {}
    void call(const Tensor& src, Tensor& dst) const override {
        engine_->cuda_reduce_to(src, dst);
    }
};

struct CUDAMatMulDispatcher final : KernelEngine::MatMulDispatcher {
    CUDAKernelEngine* engine_;
    explicit CUDAMatMulDispatcher(CUDAKernelEngine* e) : engine_(e) {}
    void call(const Tensor& a, const Tensor& b, Tensor& out) const override {
        engine_->cuda_matmul(a, b, out);
    }
};

struct CUDACopyDispatcher final : KernelEngine::CopyDispatcher {
    CUDAKernelEngine* engine_;
    explicit CUDACopyDispatcher(CUDAKernelEngine* e) : engine_(e) {}
    void call(const Tensor& src, Tensor& dst) const override {
        engine_->cuda_copy(src, dst);
    }
};

struct CUDAScaleDispatcher final : KernelEngine::ScaleDispatcher {
    CUDAKernelEngine* engine_;
    explicit CUDAScaleDispatcher(CUDAKernelEngine* e) : engine_(e) {}
    void call(Tensor& dst, double alpha) const override {
        engine_->cuda_scale(dst, alpha);
    }
};

struct CUDAAxpyDispatcher final : KernelEngine::AxpyDispatcher {
    CUDAKernelEngine* engine_;
    explicit CUDAAxpyDispatcher(CUDAKernelEngine* e) : engine_(e) {}
    void call(Tensor& dst, double alpha, const Tensor& src) const override {
        engine_->cuda_axpy(dst, alpha, src);
    }
};

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// CUDAKernelEngine constructor — registers all dispatchers
// ─────────────────────────────────────────────────────────────────────────────

CUDAKernelEngine::CUDAKernelEngine() {
    register_fill          (std::make_unique<CUDAFillDispatcher>         (this));
    register_element_read  (std::make_unique<CUDAElementReadDispatcher>  (this));
    register_bulk_host_read(std::make_unique<CUDABulkHostReadDispatcher> (this));

    // Binary element-wise: template dispatchers with contiguous/strided routing.
    // Dispatchers read engine_->default_spec_ at call time — no LaunchSpec copy.
    register_binary(KernelType::Add, std::make_unique<CUDAElementwiseBinaryKernel<AddFunctor>>(this));
    register_binary(KernelType::Sub, std::make_unique<CUDAElementwiseBinaryKernel<SubFunctor>>(this));
    register_binary(KernelType::Mul, std::make_unique<CUDAElementwiseBinaryKernel<MulFunctor>>(this));
    register_binary(KernelType::Div, std::make_unique<CUDAElementwiseBinaryKernel<DivFunctor>>(this));

    // Unary element-wise: template dispatchers with contiguous/strided routing.
    register_unary(KernelType::Neg,      std::make_unique<CUDAElementwiseUnaryKernel<NegFunctor>     >(this));
    register_unary(KernelType::Exp,      std::make_unique<CUDAElementwiseUnaryKernel<ExpFunctor>     >(this));
    register_unary(KernelType::Log,      std::make_unique<CUDAElementwiseUnaryKernel<LogFunctor>     >(this));
    register_unary(KernelType::Sqrt,     std::make_unique<CUDAElementwiseUnaryKernel<SqrtFunctor>    >(this));
    register_unary(KernelType::Relu,     std::make_unique<CUDAElementwiseUnaryKernel<ReluFunctor>    >(this));
    register_unary(KernelType::ReluMask, std::make_unique<CUDAElementwiseUnaryKernel<ReluMaskFunctor>>(this));

    // Reduction, copy, matmul, inplace: retained engine-method pattern.
    register_unary(KernelType::ReduceSum, std::make_unique<CUDAReduceSumDispatcher>(this));
    register_unary(KernelType::ReduceTo,  std::make_unique<CUDAReduceToDispatcher> (this));

    register_matmul(std::make_unique<CUDAMatMulDispatcher>(this));
    register_copy  (std::make_unique<CUDACopyDispatcher>  (this));
    register_scale (std::make_unique<CUDAScaleDispatcher> (this));
    register_axpy  (std::make_unique<CUDAAxpyDispatcher>  (this));

    // §4.2: async kernel launches. Fences are now at graph boundaries only:
    // cudaStreamSynchronize in element_read/bulk_host_read (before host reads),
    // and CUDABackend::end_backward_sync() (before Phase 5 buffer cleanup).
    default_spec_.sync_after = false;
}

} // namespace otter
