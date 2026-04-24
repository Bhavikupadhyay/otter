#include "memory/cuda_memory_manager.h"

#include <cassert>
#include <vector>

#include "cuda_check.h"
#include "otter/detail/cuda_runtime_mutex.h"
#include "otter/detail/debug_log.h"
#include "otter/detail/math_utils.h"

namespace otter {

CUDAMemoryManager::CUDAMemoryManager() = default;

CUDAMemoryManager::~CUDAMemoryManager() noexcept {
    std::vector<std::byte*> live_ptrs;
    {
        std::lock_guard<std::mutex> lock(mutex_);

        // In debug builds, a non-empty active_ means Buffers outlive the allocator.
        assert(active_.empty() &&
               "CUDAMemoryManager destroyed with live allocations");

        live_ptrs.reserve(active_.size());
        for (auto& [ptr, bytes] : active_) {
            live_ptrs.push_back(ptr);
        }
        active_.clear();
    }

    // Release all OS resources outside the mutex so shutdown cannot block
    // other allocator operations while CUDA performs its own cleanup.
    for (auto* ptr : live_ptrs) {
        std::lock_guard<std::mutex> runtime_lock(detail::cuda_runtime_mutex());
        ::cudaFree(static_cast<void*>(ptr));
    }
}

std::byte* CUDAMemoryManager::allocate(std::size_t bytes, std::size_t alignment) {
    if (bytes == 0)
        throw std::invalid_argument(
            "CUDAMemoryManager::allocate: bytes must be > 0");
    if (!detail::is_power_of_2(alignment))
        throw std::invalid_argument(
            "CUDAMemoryManager::allocate: alignment must be a power of 2");
    // cudaMallocManaged guarantees 256-byte alignment. Larger requirements
    // are not supported at this stage.
    assert(alignment <= 256 &&
           "CUDAMemoryManager::allocate: alignment > 256 not supported");

    void* raw = nullptr;
        OTTER_DBG("cuda_memory_manager: allocate begin bytes=%zu alignment=%zu", bytes, alignment);
    {
        std::lock_guard<std::mutex> runtime_lock(detail::cuda_runtime_mutex());
    OTTER_CUDA_CHECK(::cudaMallocManaged(&raw, bytes));
    }

    auto* ptr = static_cast<std::byte*>(raw);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        active_.emplace(ptr, bytes);
    }
    bytes_allocated_.fetch_add(bytes, std::memory_order_relaxed);
    OTTER_DBG("cuda_memory_manager: allocate done ptr=%p bytes=%zu", static_cast<void*>(ptr), bytes);
    return ptr;
}

void CUDAMemoryManager::free(std::byte* ptr) noexcept {
    if (!ptr) return;

    std::size_t requested = 0;
    OTTER_DBG("cuda_memory_manager: free begin ptr=%p", static_cast<void*>(ptr));
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = active_.find(ptr);
        if (it == active_.end()) {
            assert(false && "CUDAMemoryManager::free: pointer not found in active set");
            return;
        }
        requested = it->second;
        active_.erase(it);
    }

    {
        std::lock_guard<std::mutex> runtime_lock(detail::cuda_runtime_mutex());
    cudaError_t err = ::cudaFree(static_cast<void*>(ptr));
    assert(err == cudaSuccess && "CUDAMemoryManager::free: cudaFree failed");
    (void)err; // suppress unused-variable warning in release builds
    }

    bytes_allocated_.fetch_sub(requested, std::memory_order_relaxed);
    OTTER_DBG("cuda_memory_manager: free done ptr=%p bytes=%zu", static_cast<void*>(ptr), requested);
}

void CUDAMemoryManager::release_cache() noexcept {
    // No pool to flush. Synchronize device as a side-effect so callers can
    // treat release_cache() as a full device fence at checkpointing boundaries.
    OTTER_DBG("cuda_memory_manager: release_cache begin");
    std::lock_guard<std::mutex> runtime_lock(detail::cuda_runtime_mutex());
    cudaError_t err = ::cudaDeviceSynchronize();
    assert(err == cudaSuccess &&
           "CUDAMemoryManager::release_cache: cudaDeviceSynchronize failed");
    (void)err;
    OTTER_DBG("cuda_memory_manager: release_cache done");
}

Device CUDAMemoryManager::device() const noexcept {
    return Device::CUDA;
}

std::size_t CUDAMemoryManager::bytes_allocated() const noexcept {
    return bytes_allocated_.load(std::memory_order_relaxed);
}

std::size_t CUDAMemoryManager::bytes_reserved() const noexcept {
    // No pool overhead — reserved equals allocated.
    return bytes_allocated_.load(std::memory_order_relaxed);
}

} // namespace otter
