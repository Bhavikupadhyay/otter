#pragma once

#include <cstddef>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "otter/memory/memory_manager.h"
#include "otter/detail/math_utils.h"

namespace otter {

// CUDAMemoryManager — device-memory pool allocator for CUDA tensors.
//
// Two allocation paths:
//
//   Small (< kSmallAllocThreshold):
//     cudaMalloc / cudaFree directly. Never cached. Released immediately on
//     free(). bytes_reserved tracks these as live bytes only.
//
//   Large (>= kSmallAllocThreshold):
//     Segment size = max(next_power_of_2(bytes), kMinSegmentSize). Freed
//     segments enter free_pool_ keyed by segment_size for reuse. Pool is
//     searched with lower_bound(seg) — always use segment_size as the key,
//     never bytes, to match insert. cudaFree is deferred until release_cache()
//     or destructor.
//
// bytes_allocated() tracks live user bytes (both paths).
// bytes_reserved()  tracks all device memory this allocator controls:
//   - large path: cudaMalloc'd segments (live + cached pool)
//   - small path: live bytes only (freed immediately, not pooled)
//   Invariant: bytes_reserved() >= bytes_allocated() always holds.
// release_cache()   frees all pooled segments; calls cudaDeviceSynchronize
//                   first (documented checkpoint fence).
//
// Lock discipline: cuda_runtime_mutex → mutex_ (never reverse).
// CUDA API calls happen outside mutex_; pool data structure updates happen
// outside cuda_runtime_mutex. Prevents holding mutex_ during slow cudaMalloc.
class CUDAMemoryManager final : public MemoryManager {
public:
    static constexpr std::size_t kMinSegmentSize      = 2ULL * 1024ULL * 1024ULL; // 2 MB
    static constexpr std::size_t kSmallAllocThreshold = 1ULL << 20;               // 1 MB

    CUDAMemoryManager();
    ~CUDAMemoryManager() noexcept override;

    CUDAMemoryManager(const CUDAMemoryManager&)            = delete;
    CUDAMemoryManager& operator=(const CUDAMemoryManager&) = delete;
    CUDAMemoryManager(CUDAMemoryManager&&)                 = delete;
    CUDAMemoryManager& operator=(CUDAMemoryManager&&)      = delete;

    [[nodiscard]] std::byte* allocate(
        std::size_t bytes,
        std::size_t alignment = kDefaultAlignment) override;

    void free(std::byte* ptr) noexcept override;
    void release_cache() noexcept override;

    [[nodiscard]] Device      device()          const noexcept override;
    [[nodiscard]] std::size_t bytes_allocated() const noexcept override;
    [[nodiscard]] std::size_t bytes_reserved()  const noexcept override;

private:
    // Metadata for every live or pooled allocation.
    // is_large drives which cleanup path is taken in free() and the destructor.
    struct Rec {
        std::byte*  ptr;             // pointer from cudaMalloc
        std::size_t segment_size;    // bytes actually cudaMalloc'd (power-of-2 for large)
        std::size_t requested_bytes; // original caller request
        bool        is_large;
    };

    std::size_t compute_segment_size(std::size_t bytes) const noexcept {
        return std::max(detail::next_power_of_2(bytes), kMinSegmentSize);
    }

    std::unordered_map<std::byte*, Rec>  active_;    // live allocations, keyed by ptr
    std::multimap<std::size_t,     Rec>  free_pool_; // cached large segments, keyed by segment_size
    mutable std::mutex                   mutex_;     // protects active_, free_pool_, both counters
    std::size_t bytes_allocated_ = 0;               // live user bytes
    std::size_t bytes_reserved_  = 0;               // all OS-backed device bytes (live + pooled)
};

} // namespace otter
