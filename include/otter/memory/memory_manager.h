#pragma once

#include <cstddef>
#include "otter/core/device.h"

namespace otter {

class MemoryManager {
public:
    static constexpr std::size_t kDefaultAlignment = 64;

    MemoryManager(const MemoryManager&)            = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    MemoryManager(MemoryManager&&)                 = delete;
    MemoryManager& operator=(MemoryManager&&)      = delete;

    virtual ~MemoryManager() = default;

    // Allocate at least `bytes` bytes aligned to `alignment`.
    // Returns std::byte* — untyped storage. Callers cast to their numeric
    // type at the point of use (Buffer does this via reinterpret_cast).
    // Throws std::invalid_argument if bytes == 0 or alignment is not
    // a power of 2. Throws std::bad_alloc if the allocation fails.
    [[nodiscard]] virtual std::byte* allocate(
        std::size_t bytes,
        std::size_t alignment = kDefaultAlignment) = 0;

    // Return a previously allocated pointer to the manager.
    // ptr must have been returned by this manager's allocate(). noexcept
    // so destructors can call it safely.
    virtual void free(std::byte* ptr) noexcept = 0;

    // Return all cached (freed but not yet unmapped) segments to the OS.
    // Analogous to torch.cuda.empty_cache(). Call at natural checkpointing
    // boundaries. Re-allocating after a release incurs full mmap cost.
    virtual void release_cache() noexcept = 0;

    // Copy `bytes` bytes from host pointer `src` into device allocation `dst`.
    // `dst` must have been returned by this manager's allocate().
    // CPU backends: plain memcpy. CUDA backends: cudaMemcpy(HostToDevice).
    virtual void copy_from_host(void* dst, const void* src, std::size_t bytes) = 0;

    // Zero-fill `bytes` bytes at device allocation `dst`.
    // `dst` must have been returned by this manager's allocate().
    virtual void zero_fill(void* dst, std::size_t bytes) = 0;

    [[nodiscard]] virtual Device      device()          const noexcept = 0;
    [[nodiscard]] virtual std::size_t bytes_allocated() const noexcept = 0;
    [[nodiscard]] virtual std::size_t bytes_reserved()  const noexcept = 0;

protected:
    MemoryManager() = default;
};

} // namespace otter
