#pragma once

#include <memory>
#include "otter/core/device.h"
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

    ~Backend() = default;

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

private:
    std::unique_ptr<MemoryManager> mm_;
    std::unique_ptr<KernelEngine>  ke_;
};

} // namespace otter
