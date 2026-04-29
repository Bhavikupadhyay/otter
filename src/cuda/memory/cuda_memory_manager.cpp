#include "cuda/memory/cuda_memory_manager.h"

#include <cassert>
#include <new>
#include <vector>

#include "cuda/cuda_check.h"
#include "otter/detail/debug_log.h"
#include "otter/detail/math_utils.h"

namespace otter {

CUDAMemoryManager::CUDAMemoryManager() = default;

CUDAMemoryManager::~CUDAMemoryManager() noexcept {
    std::vector<void*> live_ptrs;
    std::vector<void*> pool_ptrs;
    {
        std::lock_guard<std::mutex> lock(mutex_);

        // In debug builds, a non-empty active_ means Buffers outlive the allocator.
        assert(active_.empty() &&
               "CUDAMemoryManager destroyed with live allocations");

        live_ptrs.reserve(active_.size());
        for (auto& [ptr, rec] : active_) {
            live_ptrs.push_back(static_cast<void*>(ptr));
        }
        active_.clear();

        pool_ptrs.reserve(free_pool_.size());
        for (auto& [size, rec] : free_pool_) {
            pool_ptrs.push_back(static_cast<void*>(rec.ptr));
        }
        free_pool_.clear();
    }

    // Release all device memory outside mutex_ so shutdown cannot block other
    // allocator operations. cudaFree is thread-safe; no external lock needed.
    for (void* p : live_ptrs) ::cudaFree(p);
    for (void* p : pool_ptrs) ::cudaFree(p);
}

std::byte* CUDAMemoryManager::allocate(std::size_t bytes, std::size_t alignment) {
    if (bytes == 0)
        throw std::invalid_argument(
            "CUDAMemoryManager::allocate: bytes must be > 0");
    if (!detail::is_power_of_2(alignment))
        throw std::invalid_argument(
            "CUDAMemoryManager::allocate: alignment must be a power of 2");
    // cudaMalloc guarantees at least 256-byte alignment. Larger requirements
    // are not supported at this stage.
    if (alignment > 256) {
        throw std::invalid_argument(
            "CUDAMemoryManager::allocate: alignment > 256 not supported");
    }

    OTTER_DBG("cuda_memory_manager: allocate begin bytes=%zu alignment=%zu", bytes, alignment);

    // ── Small path: direct cudaMalloc, never pooled ───────────────────────────
    if (bytes < kSmallAllocThreshold) {
        void* raw = nullptr;
        const cudaError_t err = ::cudaMalloc(&raw, bytes);
        if (err == cudaErrorMemoryAllocation) throw std::bad_alloc();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("CUDA error: ") + ::cudaGetErrorString(err));

        auto* ptr = static_cast<std::byte*>(raw);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            active_.emplace(ptr, Rec{ptr, bytes, bytes, false});
            bytes_allocated_ += bytes;
            bytes_reserved_  += bytes;
        }
        OTTER_DBG("cuda_memory_manager: allocate small done ptr=%p bytes=%zu",
                  static_cast<void*>(ptr), bytes);
        return ptr;
    }

    // ── Large path: try pool first ────────────────────────────────────────────
    const std::size_t seg = compute_segment_size(bytes);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = free_pool_.lower_bound(seg);
        if (it != free_pool_.end()) {
            Rec rec          = it->second;
            free_pool_.erase(it);
            rec.requested_bytes = bytes;
            active_.emplace(rec.ptr, rec);
            bytes_allocated_ += bytes;
            // bytes_reserved_ unchanged: segment stays in device memory
            OTTER_DBG("cuda_memory_manager: allocate pool hit ptr=%p seg=%zu bytes=%zu",
                      static_cast<void*>(rec.ptr), rec.segment_size, bytes);
            return rec.ptr;
        }
    }

    // Pool miss: allocate a new segment from the device.
    void* raw = nullptr;
    cudaError_t err = ::cudaMalloc(&raw, seg);

    // On OOM: evict the entire free pool and retry once before propagating
    // the error. Matches release_cache() discipline: drain under mutex_, then
    // sync + free outside to avoid holding a lock during slow driver calls.
    if (err == cudaErrorMemoryAllocation) {
        OTTER_DBG("cuda_memory_manager: OOM on seg=%zu — evicting pool and retrying", seg);
        std::vector<void*> evicted;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            evicted.reserve(free_pool_.size());
            for (auto& [sz, rec] : free_pool_) {
                bytes_reserved_ -= rec.segment_size;
                evicted.push_back(static_cast<void*>(rec.ptr));
            }
            free_pool_.clear();
        }
        cudaError_t sync_err = ::cudaDeviceSynchronize();
        assert(sync_err == cudaSuccess &&
               "CUDAMemoryManager::allocate: cudaDeviceSynchronize failed during eviction");
        (void)sync_err;
        for (void* p : evicted) ::cudaFree(p);
        err = ::cudaMalloc(&raw, seg);  // single retry
    }

    if (err == cudaErrorMemoryAllocation) throw std::bad_alloc();
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("CUDA error: ") + ::cudaGetErrorString(err));

    auto* ptr = static_cast<std::byte*>(raw);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        active_.emplace(ptr, Rec{ptr, seg, bytes, true});
        bytes_allocated_ += bytes;
        bytes_reserved_  += seg;
    }
    OTTER_DBG("cuda_memory_manager: allocate large done ptr=%p seg=%zu bytes=%zu",
              static_cast<void*>(ptr), seg, bytes);
    return ptr;
}

void CUDAMemoryManager::free(std::byte* ptr) noexcept {
    if (!ptr) return;

    bool is_large = false;
    Rec  rec{};
    OTTER_DBG("cuda_memory_manager: free begin ptr=%p", static_cast<void*>(ptr));
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = active_.find(ptr);
        if (it == active_.end()) {
            assert(false && "CUDAMemoryManager::free: pointer not found in active set");
            return;
        }
        rec      = it->second;
        is_large = rec.is_large;
        active_.erase(it);
        bytes_allocated_ -= rec.requested_bytes;

        if (is_large) {
            // Defer cudaFree: return segment to the pool.
            free_pool_.emplace(rec.segment_size, rec);
            // bytes_reserved_ unchanged: segment remains device-backed in the pool.
        } else {
            bytes_reserved_ -= rec.segment_size;
        }
    }

    if (!is_large) {
        // cudaFree is thread-safe; no external lock needed.
        cudaError_t e = ::cudaFree(static_cast<void*>(ptr));
        assert(e == cudaSuccess && "CUDAMemoryManager::free: cudaFree failed");
        (void)e;
    }
    OTTER_DBG("cuda_memory_manager: free done ptr=%p is_large=%d",
              static_cast<void*>(ptr), static_cast<int>(is_large));
}

void CUDAMemoryManager::release_cache() noexcept {
    OTTER_DBG("cuda_memory_manager: release_cache begin");

    std::vector<void*> to_free;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        to_free.reserve(free_pool_.size());
        for (auto& [size, rec] : free_pool_) {
            bytes_reserved_ -= rec.segment_size;
            to_free.push_back(static_cast<void*>(rec.ptr));
        }
        free_pool_.clear();
    }

    // Synchronize device before freeing — documented as a full checkpoint fence.
    // cudaDeviceSynchronize and cudaFree are thread-safe; no external lock needed.
    cudaError_t err = ::cudaDeviceSynchronize();
    assert(err == cudaSuccess &&
           "CUDAMemoryManager::release_cache: cudaDeviceSynchronize failed");
    (void)err;
    for (void* p : to_free) {
        err = ::cudaFree(p);
        assert(err == cudaSuccess &&
               "CUDAMemoryManager::release_cache: cudaFree failed");
        (void)err;
    }
    OTTER_DBG("cuda_memory_manager: release_cache done freed=%zu", to_free.size());
}

Device CUDAMemoryManager::device() const noexcept {
    return Device::CUDA;
}

std::size_t CUDAMemoryManager::bytes_allocated() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return bytes_allocated_;
}

std::size_t CUDAMemoryManager::bytes_reserved() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return bytes_reserved_;
}

} // namespace otter
