#include "cpu/memory/cpu_memory_manager.h"

#include <cassert>
#include <cstring>
#include <new>
#include <stdexcept>

#include <sys/mman.h>
#include <unistd.h>

namespace otter {

CPUMemoryManager::CPUMemoryManager(std::size_t min_seg)
    : min_segment_size_(min_seg)
{
    if (!detail::is_power_of_2(min_segment_size_) ||
        min_segment_size_ < static_cast<std::size_t>(::sysconf(_SC_PAGESIZE)))
        throw std::invalid_argument(
            "CPUMemoryManager: min_segment_size must be a power of 2 "
            "and >= the system page size");
}

CPUMemoryManager::~CPUMemoryManager() noexcept {
    // Lock for the same reason allocate()/free() lock: if a Backend is ever
    // heap-allocated and destroyed concurrently with a final Buffer::~Buffer()
    // calling free(), the destructor must not race on active_ or free_pool_.
    // For the current singleton pattern this is defensive, but correct.
    std::lock_guard<std::mutex> lock(mutex_);

    // In debug builds, a non-empty active_ means the caller has live
    // Buffers that outlive the allocator — a lifetime contract violation.
    assert(active_.empty() &&
           "CPUMemoryManager destroyed with live allocations");

    // Release all OS resources regardless, so we don't leak in release builds.
    for (auto& [ptr, rec] : active_) {
        if (rec.is_small) ::free(rec.base_ptr);
        else              ::munmap(rec.base_ptr, rec.mapped_size);
    }
    for (auto& [size, rec] : free_pool_) {
        ::munmap(rec.base_ptr, rec.mapped_size);
    }
}

std::byte* CPUMemoryManager::allocate(std::size_t bytes, std::size_t alignment) {
    if (bytes == 0)
        throw std::invalid_argument("CPUMemoryManager::allocate: bytes must be > 0");
    if (!detail::is_power_of_2(alignment))
        throw std::invalid_argument(
            "CPUMemoryManager::allocate: alignment must be a power of 2");

    std::lock_guard<std::mutex> lock(mutex_);

    // ── Small path: posix_memalign ───────────────────────────────────────────
    // Avoids burning a 2 MB mmap segment for small tensors.
    // Small allocations are never cached — freed immediately on free().
    if (bytes < kSmallAllocThreshold) {
        void* raw = nullptr;
        const std::size_t actual_align = std::max(alignment, sizeof(void*));
        if (::posix_memalign(&raw, actual_align, bytes) != 0)
            throw std::bad_alloc{};
        auto* ptr = static_cast<std::byte*>(raw);
        std::memset(ptr, 0, bytes); // posix_memalign does not zero-initialise
        Rec rec{ptr, ptr, bytes, bytes, alignment, /*is_small=*/true};
        active_.emplace(ptr, rec);
        bytes_allocated_ += bytes;
        bytes_reserved_  += bytes;  // small allocs count toward reserved too
        return ptr;
    }

    // ── Large path: mmap pool ────────────────────────────────────────────────
    // Segments are power-of-2 sized (minimum kMinSegmentSize).
    // Pool is keyed by mapped_size. Search uses lower_bound(seg) — not
    // lower_bound(bytes) — so the search key matches what was inserted.
    const std::size_t extra = (alignment > 4096) ? (alignment - 1) : 0;
    const std::size_t seg   = compute_segment_size(bytes + extra);

    for (auto it = free_pool_.lower_bound(seg); it != free_pool_.end(); ++it) {
        Rec& cached = it->second;
        auto* aligned = static_cast<std::byte*>(
            detail::align_ptr(cached.base_ptr, alignment));
        if (aligned + bytes <= cached.base_ptr + cached.mapped_size) {
            Rec rec = cached;
            rec.user_ptr        = aligned;
            rec.requested_bytes = bytes;
            rec.alignment       = alignment;
            rec.is_small        = false;
            active_.emplace(aligned, rec);
            bytes_allocated_ += bytes;
            free_pool_.erase(it);
            return aligned;
        }
    }

    // No suitable cached segment — map a fresh one.
    void* raw = ::mmap(nullptr, seg,
                       PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS,
                       -1, 0);
    if (raw == MAP_FAILED) throw std::bad_alloc{};

    // TODO(perf): madvise(base, seg, MADV_HUGEPAGE) once pipeline is stable.

    auto* base    = static_cast<std::byte*>(raw);
    auto* aligned = static_cast<std::byte*>(detail::align_ptr(base, alignment));
    Rec rec{base, aligned, seg, bytes, alignment, /*is_small=*/false};
    active_.emplace(aligned, rec);
    bytes_allocated_ += bytes;
    bytes_reserved_  += seg;
    return aligned;
}

void CPUMemoryManager::free(std::byte* ptr) noexcept {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = active_.find(ptr);
    if (it == active_.end()) {
        assert(false && "CPUMemoryManager::free: pointer not found in active set");
        return;
    }

    Rec rec = it->second;
    active_.erase(it);
    bytes_allocated_ -= rec.requested_bytes;

    if (rec.is_small) {
        bytes_reserved_ -= rec.requested_bytes;
        ::free(rec.base_ptr);
    } else {
        free_pool_.emplace(rec.mapped_size, rec);
    }
}

void CPUMemoryManager::release_cache() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [size, rec] : free_pool_) {
        bytes_reserved_ -= rec.mapped_size;
        ::munmap(rec.base_ptr, rec.mapped_size);
    }
    free_pool_.clear();
}

void CPUMemoryManager::copy_from_host(void* dst, const void* src, std::size_t bytes) {
    std::memcpy(dst, src, bytes);
}

void CPUMemoryManager::zero_fill(void* dst, std::size_t bytes) {
    std::memset(dst, 0, bytes);
}

std::size_t CPUMemoryManager::bytes_allocated() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return bytes_allocated_;
}

std::size_t CPUMemoryManager::bytes_reserved() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return bytes_reserved_;
}

} // namespace otter
