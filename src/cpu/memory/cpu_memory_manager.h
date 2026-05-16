#pragma once

#include <cstddef>
#include <map>
#include <mutex>
#include <unordered_map>

#include "otter/memory/memory_manager.h"
#include "otter/detail/math_utils.h"

namespace otter {

// CPUMemoryManager — mmap pool allocator for large tensors, posix_memalign
// for small ones.
//
// Two allocation paths:
//
//   Small (< kSmallAllocThreshold):
//     posix_memalign / ::free. Never cached. Freed immediately on free().
//     memset to zero on allocation (posix_memalign does not guarantee it).
//
//   Large (>= kSmallAllocThreshold):
//     mmap with MAP_PRIVATE|MAP_ANONYMOUS. Segments are power-of-2 sized,
//     minimum kMinSegmentSize (2 MB). Freed segments return to free_pool_
//     for reuse. Pool is keyed by mapped_size — search always uses
//     lower_bound(seg), never lower_bound(bytes), to match the insert key.
//
// allocate() returns std::byte* — untyped storage with C++17 aliasing
// exemption ([basic.lval]). Casting to double* or float* via reinterpret_cast
// is well-defined provided the alignment contract is met (enforced by the
// alignment parameter, defaulting to 64 bytes).
//
// bytes_allocated() tracks live user bytes (both paths).
// bytes_reserved()  tracks all OS-backed memory this allocator controls:
//   - large path: mmap'd pages (live + cached pool)
//   - small path: posix_memalign'd bytes (live only; not pooled)
//   Invariant: bytes_reserved() >= bytes_allocated() always holds.
// release_cache()   munmaps all free_pool_ segments; allocator stays functional.
class CPUMemoryManager final : public MemoryManager {
public:
    static constexpr std::size_t kMinSegmentSize      = 2ULL * 1024ULL * 1024ULL; // 2 MB
    static constexpr std::size_t kSmallAllocThreshold = 1ULL << 20;               // 1 MB

    explicit CPUMemoryManager(std::size_t min_seg = kMinSegmentSize);

    ~CPUMemoryManager() noexcept override;

    CPUMemoryManager(const CPUMemoryManager&)            = delete;
    CPUMemoryManager& operator=(const CPUMemoryManager&) = delete;
    CPUMemoryManager(CPUMemoryManager&&)                 = delete;
    CPUMemoryManager& operator=(CPUMemoryManager&&)      = delete;

    [[nodiscard]] std::byte* allocate(
        std::size_t bytes,
        std::size_t alignment = kDefaultAlignment) override;

    void free(std::byte* ptr) noexcept override;

    void release_cache() noexcept override;

    void copy_from_host(void* dst, const void* src, std::size_t bytes) override;
    void zero_fill(void* dst, std::size_t bytes) override;

    Device      device()          const noexcept override { return Device::CPU; }
    std::size_t bytes_allocated() const noexcept override;
    std::size_t bytes_reserved()  const noexcept override;

private:
    // Metadata for every live or cached allocation.
    // is_small drives which cleanup path is taken in free() and the destructor.
    struct Rec {
        std::byte*  base_ptr;        // original pointer from mmap / posix_memalign
        std::byte*  user_ptr;        // aligned pointer handed to the caller
        std::size_t mapped_size;     // total bytes mmap'd (segment size); unused for small
        std::size_t requested_bytes; // what the caller asked for
        std::size_t alignment;
        bool        is_small;
    };

    std::size_t compute_segment_size(std::size_t bytes) const noexcept {
        return std::max(detail::next_power_of_2(bytes), min_segment_size_);
    }

    std::unordered_map<std::byte*, Rec> active_;    // live allocations, keyed by user_ptr
    std::multimap<std::size_t,     Rec> free_pool_; // cached segments, keyed by mapped_size
    mutable std::mutex                  mutex_;
    std::size_t min_segment_size_;
    std::size_t bytes_allocated_ = 0;
    std::size_t bytes_reserved_  = 0;
};

} // namespace otter
