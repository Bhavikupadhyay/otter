#include "cuda_kernel_engine.h"

#include "otter/tensor.h"

#include <memory>

namespace otter {

namespace {

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

} // namespace

CUDAKernelEngine::CUDAKernelEngine() {
    register_fill           (std::make_unique<CUDAFillDispatcher>           (this));
    register_element_read   (std::make_unique<CUDAElementReadDispatcher>    (this));
    register_bulk_host_read (std::make_unique<CUDABulkHostReadDispatcher>   (this));
}

} // namespace otter
