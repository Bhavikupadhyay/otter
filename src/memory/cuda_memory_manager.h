#pragma once

#include <atomic>
#include <cstddef>
#include <mutex>
#include <unordered_map>

#include "otter/memory/memory_manager.h"

namespace otter {

// CUDAMemoryManager — unified-memory allocator for CUDA tensors.
//
// Uses cudaMallocManaged: returned pointers are accessible from both host and
// device. The CUDA driver migrates pages on demand, so Tensor::at() works
// without explicit device→host copies. No pooling — each allocate/free maps
// directly to a cudaMallocManaged/cudaFree pair.
//
// bytes_allocated() tracks live user bytes.
// bytes_reserved()  == bytes_allocated() — no pool, no overhead.
// release_cache()   calls cudaDeviceSynchronize() then returns — no pool to
//                   flush. Synchronizes pending device work as a side-effect.
//
// Alignment: cudaMallocManaged guarantees 256-byte alignment, which satisfies
// kDefaultAlignment (64). The alignment parameter is asserted <= 256 and
// otherwise ignored.
class CUDAMemoryManager final : public MemoryManager {
public:
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
    mutable std::mutex                          mutex_;
    std::unordered_map<std::byte*, std::size_t> active_;          // ptr -> requested bytes
    std::atomic<std::size_t>                    bytes_allocated_{0};
};

} // namespace otter
