#pragma once

#include <memory>
#include "otter/core/device.h"
#include "otter/core/stream.h"
#include "otter/memory/memory_manager.h"
#include "otter/kernel/kernel_engine.h"

namespace otter {

// Backend — pairs a MemoryManager with a KernelEngine for one device.
//
// Owns both resources. Users obtain a Backend via a factory such as
// cpu_backend() rather than constructing one directly. A CUDA backend
// would provide cuda_backend() following the same pattern.
//
// Backend* is what Tensor stores (non-owning). The Backend must outlive
// all Tensors allocated through it.
class Backend {
public:
    explicit Backend(std::unique_ptr<MemoryManager> mm,
                     std::unique_ptr<KernelEngine>  ke);

    // Virtual destructor: Backend now has virtual methods (default_stream),
    // so CUDA subclasses need virtual dispatch for correct cleanup.
    virtual ~Backend() = default;

    Backend(const Backend&)            = delete;
    Backend& operator=(const Backend&) = delete;
    Backend(Backend&&)                 = delete;
    Backend& operator=(Backend&&)      = delete;

    [[nodiscard]] MemoryManager*       memory_manager()       noexcept { return mm_.get(); }
    [[nodiscard]] const MemoryManager* memory_manager() const noexcept { return mm_.get(); }

    [[nodiscard]] KernelEngine*        kernel_engine()        noexcept { return ke_.get(); }
    [[nodiscard]] const KernelEngine*  kernel_engine()  const noexcept { return ke_.get(); }

    // Device is determined by the MemoryManager; both must agree.
    [[nodiscard]] Device device() const noexcept { return mm_->device(); }

    // Returns the default execution stream for this backend.
    //
    // CPU: always nullptr — the CPU backend is synchronous; there is no stream.
    // CUDA: returns the per-device default stream (cudaStreamDefault equivalent).
    //
    // Non-pure virtual with a nullptr default so the existing cpu_backend()
    // factory (which constructs a plain Backend, not a subclass) compiles
    // unchanged. A CUDA backend subclass overrides this to return a real stream.
    [[nodiscard]] virtual Stream* default_stream() noexcept { return nullptr; }

    // Blocks until all device-side work submitted through this backend has completed.
    // Called by GraphExecutor between Phase 4 (workers drain) and Phase 5 (clear_saved)
    // so that no Buffer is freed while a GPU kernel is still writing it.
    // CPU default is a no-op. CUDA override calls cudaStreamSynchronize.
    virtual void end_backward_sync() noexcept {}

private:
    std::unique_ptr<MemoryManager> mm_;
    std::unique_ptr<KernelEngine>  ke_;
};

} // namespace otter
