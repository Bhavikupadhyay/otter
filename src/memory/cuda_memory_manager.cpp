#include "memory/cuda_memory_manager.h"

#include <cassert>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "otter/detail/math_utils.h"

// Wrap every CUDA call. Throws on error — callers that are noexcept use the
// raw CUDA API directly and check via assert instead.
#define OTTER_CUDA_CHECK(call)                                              \
    do {                                                                    \
        cudaError_t _err = (call);                                          \
        if (_err != cudaSuccess)                                            \
            throw std::runtime_error(                                       \
                std::string("CUDA error: ") + ::cudaGetErrorString(_err));  \
    } while (0)

namespace otter {

CUDAMemoryManager::CUDAMemoryManager() = default;

CUDAMemoryManager::~CUDAMemoryManager() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);

    // In debug builds, a non-empty active_ means Buffers outlive the allocator.
    assert(active_.empty() &&
           "CUDAMemoryManager destroyed with live allocations");

    // Release all OS resources regardless, so we don't leak in release builds.
    for (auto& [ptr, bytes] : active_) {
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
    OTTER_CUDA_CHECK(::cudaMallocManaged(&raw, bytes));

    auto* ptr = static_cast<std::byte*>(raw);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        active_.emplace(ptr, bytes);
    }
    bytes_allocated_.fetch_add(bytes, std::memory_order_relaxed);
    return ptr;
}

void CUDAMemoryManager::free(std::byte* ptr) noexcept {
    if (!ptr) return;

    std::size_t requested = 0;
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

    cudaError_t err = ::cudaFree(static_cast<void*>(ptr));
    assert(err == cudaSuccess && "CUDAMemoryManager::free: cudaFree failed");
    (void)err; // suppress unused-variable warning in release builds

    bytes_allocated_.fetch_sub(requested, std::memory_order_relaxed);
}

void CUDAMemoryManager::release_cache() noexcept {
    // No pool to flush. Synchronize device as a side-effect so callers can
    // treat release_cache() as a full device fence at checkpointing boundaries.
    cudaError_t err = ::cudaDeviceSynchronize();
    assert(err == cudaSuccess &&
           "CUDAMemoryManager::release_cache: cudaDeviceSynchronize failed");
    (void)err;
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
