#pragma once

#include <cstddef>
#include <type_traits>
#include "otter/core/device.h"
#include "otter/core/passkey.h"
#include "otter/memory/memory_manager.h"

namespace otter {

// Forward declaration — Passkey gate only; no full definition needed here.
class KernelEngine;

// Buffer — RAII wrapper over a single OS allocation.
//
// Owns exactly one allocation on exactly one device. Non-copyable,
// non-movable. shared_ptr<Buffer> is how multiple Tensors share one
// allocation (view semantics).
//
// Raw data access is gated exclusively by Passkey<KernelEngine>. Only
// KernelEngine subclasses can reach the underlying pointer — everything
// else goes through a dispatch_* virtual. This enforces the device
// abstraction: a CUDA buffer's pointer is a device address and must
// never be dereferenced on the host outside of a kernel dispatch.
//
// Every raw-pointer site is auditable with: grep Passkey<KernelEngine>
class Buffer {
public:
    // Allocates `bytes` bytes via `mm->allocate()`.
    // If `init_data` is non-null, copies it into the new allocation.
    // Otherwise zero-initialises (pool segments may not be zero on reuse).
    explicit Buffer(std::size_t bytes, MemoryManager* mm,
                    const void* init_data = nullptr);

    ~Buffer() noexcept;

    Buffer(const Buffer&)            = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&&)                 = delete;
    Buffer& operator=(Buffer&&)      = delete;

    std::size_t size()   const noexcept { return size_; }
    Device      device() const noexcept { return device_; }

    // Read access — kernel dispatch only.
    // T must be a numeric scalar type (double, float).
    template<typename T>
    [[nodiscard]] const T* data(Passkey<KernelEngine>) const noexcept {
        static_assert(std::is_arithmetic_v<T>,
                      "Buffer::data: T must be a numeric scalar type");
        return reinterpret_cast<const T*>(data_);
    }

    // Write access — kernel dispatch only.
    // T must be a numeric scalar type (double, float).
    template<typename T>
    [[nodiscard]] T* mutable_data(Passkey<KernelEngine>) noexcept {
        static_assert(std::is_arithmetic_v<T>,
                      "Buffer::mutable_data: T must be a numeric scalar type");
        return reinterpret_cast<T*>(data_);
    }

private:
    std::byte*     data_;
    std::size_t    size_;
    Device         device_;
    MemoryManager* allocator_;  // non-owning; must outlive Buffer
};

} // namespace otter
