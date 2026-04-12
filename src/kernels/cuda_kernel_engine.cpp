#include "cuda_kernel_engine.h"

#include "otter/tensor.h"

#include <memory>

namespace otter {

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Existing dispatchers (fill / element_read / bulk_host_read)
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

// ─────────────────────────────────────────────────────────────────────────────
// Binary element-wise dispatchers
// ─────────────────────────────────────────────────────────────────────────────

struct CUDAAddDispatcher final : KernelEngine::BinaryDispatcher {
    CUDAKernelEngine* engine_;
    explicit CUDAAddDispatcher(CUDAKernelEngine* e) : engine_(e) {}
    void call(const Tensor& a, const Tensor& b, Tensor& out) const override {
        engine_->cuda_add(a, b, out);
    }
};

struct CUDASubDispatcher final : KernelEngine::BinaryDispatcher {
    CUDAKernelEngine* engine_;
    explicit CUDASubDispatcher(CUDAKernelEngine* e) : engine_(e) {}
    void call(const Tensor& a, const Tensor& b, Tensor& out) const override {
        engine_->cuda_sub(a, b, out);
    }
};

struct CUDAMulDispatcher final : KernelEngine::BinaryDispatcher {
    CUDAKernelEngine* engine_;
    explicit CUDAMulDispatcher(CUDAKernelEngine* e) : engine_(e) {}
    void call(const Tensor& a, const Tensor& b, Tensor& out) const override {
        engine_->cuda_mul(a, b, out);
    }
};

struct CUDADivDispatcher final : KernelEngine::BinaryDispatcher {
    CUDAKernelEngine* engine_;
    explicit CUDADivDispatcher(CUDAKernelEngine* e) : engine_(e) {}
    void call(const Tensor& a, const Tensor& b, Tensor& out) const override {
        engine_->cuda_div(a, b, out);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Unary element-wise dispatchers
// ─────────────────────────────────────────────────────────────────────────────

struct CUDANegDispatcher final : KernelEngine::UnaryDispatcher {
    CUDAKernelEngine* engine_;
    explicit CUDANegDispatcher(CUDAKernelEngine* e) : engine_(e) {}
    void call(const Tensor& a, Tensor& out) const override {
        engine_->cuda_neg(a, out);
    }
};

struct CUDAExpDispatcher final : KernelEngine::UnaryDispatcher {
    CUDAKernelEngine* engine_;
    explicit CUDAExpDispatcher(CUDAKernelEngine* e) : engine_(e) {}
    void call(const Tensor& a, Tensor& out) const override {
        engine_->cuda_exp(a, out);
    }
};

struct CUDALogDispatcher final : KernelEngine::UnaryDispatcher {
    CUDAKernelEngine* engine_;
    explicit CUDALogDispatcher(CUDAKernelEngine* e) : engine_(e) {}
    void call(const Tensor& a, Tensor& out) const override {
        engine_->cuda_log(a, out);
    }
};

struct CUDASqrtDispatcher final : KernelEngine::UnaryDispatcher {
    CUDAKernelEngine* engine_;
    explicit CUDASqrtDispatcher(CUDAKernelEngine* e) : engine_(e) {}
    void call(const Tensor& a, Tensor& out) const override {
        engine_->cuda_sqrt(a, out);
    }
};

struct CUDAReluDispatcher final : KernelEngine::UnaryDispatcher {
    CUDAKernelEngine* engine_;
    explicit CUDAReluDispatcher(CUDAKernelEngine* e) : engine_(e) {}
    void call(const Tensor& a, Tensor& out) const override {
        engine_->cuda_relu(a, out);
    }
};

struct CUDAReluMaskDispatcher final : KernelEngine::UnaryDispatcher {
    CUDAKernelEngine* engine_;
    explicit CUDAReluMaskDispatcher(CUDAKernelEngine* e) : engine_(e) {}
    void call(const Tensor& a, Tensor& out) const override {
        engine_->cuda_relu_mask(a, out);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Reduction dispatchers (use UnaryDispatcher signature)
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// MatMul dispatcher
// ─────────────────────────────────────────────────────────────────────────────

struct CUDAMatMulDispatcher final : KernelEngine::MatMulDispatcher {
    CUDAKernelEngine* engine_;
    explicit CUDAMatMulDispatcher(CUDAKernelEngine* e) : engine_(e) {}
    void call(const Tensor& a, const Tensor& b, Tensor& out) const override {
        engine_->cuda_matmul(a, b, out);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Copy dispatcher
// ─────────────────────────────────────────────────────────────────────────────

struct CUDACopyDispatcher final : KernelEngine::CopyDispatcher {
    CUDAKernelEngine* engine_;
    explicit CUDACopyDispatcher(CUDAKernelEngine* e) : engine_(e) {}
    void call(const Tensor& src, Tensor& dst) const override {
        engine_->cuda_copy(src, dst);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// In-place dispatchers
// ─────────────────────────────────────────────────────────────────────────────

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
    register_fill           (std::make_unique<CUDAFillDispatcher>           (this));
    register_element_read   (std::make_unique<CUDAElementReadDispatcher>    (this));
    register_bulk_host_read (std::make_unique<CUDABulkHostReadDispatcher>   (this));

    register_binary(KernelType::Add, std::make_unique<CUDAAddDispatcher>(this));
    register_binary(KernelType::Sub, std::make_unique<CUDASubDispatcher>(this));
    register_binary(KernelType::Mul, std::make_unique<CUDAMulDispatcher>(this));
    register_binary(KernelType::Div, std::make_unique<CUDADivDispatcher>(this));

    register_unary(KernelType::Neg,       std::make_unique<CUDANegDispatcher>      (this));
    register_unary(KernelType::Exp,       std::make_unique<CUDAExpDispatcher>      (this));
    register_unary(KernelType::Log,       std::make_unique<CUDALogDispatcher>      (this));
    register_unary(KernelType::Sqrt,      std::make_unique<CUDASqrtDispatcher>     (this));
    register_unary(KernelType::Relu,      std::make_unique<CUDAReluDispatcher>     (this));
    register_unary(KernelType::ReluMask,  std::make_unique<CUDAReluMaskDispatcher> (this));
    register_unary(KernelType::ReduceSum, std::make_unique<CUDAReduceSumDispatcher>(this));
    register_unary(KernelType::ReduceTo,  std::make_unique<CUDAReduceToDispatcher> (this));

    register_matmul(std::make_unique<CUDAMatMulDispatcher>(this));
    register_copy  (std::make_unique<CUDACopyDispatcher>  (this));
    register_scale (std::make_unique<CUDAScaleDispatcher> (this));
    register_axpy  (std::make_unique<CUDAAxpyDispatcher>  (this));
}

} // namespace otter
