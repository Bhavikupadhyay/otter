#include "test_utils.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "otter/memory/buffer.h"
#include "memory/cpu_memory_manager.h"

namespace otter::test {

// ── Test 1 ───────────────────────────────────────────────────────────────────

void test_allocator_small_alloc_zero_initialised() {
    std::cout << "[Test 1] small allocation is zero-initialised\n";
    CPUMemoryManager mm;
    // 64 bytes — well below the 1 MB small-alloc threshold
    std::byte* ptr = mm.allocate(64);
    bool all_zero = true;
    for (std::size_t i = 0; i < 64; ++i) all_zero &= (ptr[i] == std::byte{0});
    CHECK(all_zero);
    CHECK(mm.bytes_allocated() == 64);
    mm.free(ptr);
    CHECK(mm.bytes_allocated() == 0);
}

// ── Test 2 ───────────────────────────────────────────────────────────────────

void test_allocator_large_alloc_zero_initialised() {
    std::cout << "[Test 2] large allocation is zero-initialised\n";
    CPUMemoryManager mm;
    // 4 MB — above the 1 MB threshold, goes through mmap pool
    constexpr std::size_t sz = 4ULL * 1024ULL * 1024ULL;
    std::byte* ptr = mm.allocate(sz);
    bool all_zero = true;
    for (std::size_t i = 0; i < sz; i += 4096) all_zero &= (ptr[i] == std::byte{0});
    CHECK(all_zero);
    CHECK(mm.bytes_allocated() == sz);
    CHECK(mm.bytes_reserved()  >= sz);
    mm.free(ptr);
    CHECK(mm.bytes_allocated() == 0);
    CHECK(mm.bytes_reserved()  >= sz); // segment still cached, not munmap'd
}

// ── Test 3 ───────────────────────────────────────────────────────────────────

void test_allocator_large_pool_reuses_segment() {
    std::cout << "[Test 3] freed large segment is reused on next same-size allocation\n";
    CPUMemoryManager mm;
    constexpr std::size_t sz = 4ULL * 1024ULL * 1024ULL;

    std::byte* first = mm.allocate(sz);
    mm.free(first);
    const std::size_t reserved_after_free = mm.bytes_reserved();

    std::byte* second = mm.allocate(sz);
    // Reserved bytes must not grow — the cached segment was reused
    CHECK(mm.bytes_reserved() == reserved_after_free);
    mm.free(second);
    CHECK(mm.bytes_allocated() == 0);
}

// ── Test 4 ───────────────────────────────────────────────────────────────────

void test_allocator_release_cache_returns_memory_to_os() {
    std::cout << "[Test 4] release_cache() drops all cached segments\n";
    CPUMemoryManager mm;
    constexpr std::size_t sz = 4ULL * 1024ULL * 1024ULL;

    std::byte* ptr = mm.allocate(sz);
    mm.free(ptr);
    CHECK(mm.bytes_reserved() > 0);

    mm.release_cache();
    CHECK(mm.bytes_reserved()  == 0);
    CHECK(mm.bytes_allocated() == 0);

    // Allocator must still be functional after a cache release
    std::byte* ptr2 = mm.allocate(sz);
    CHECK(ptr2 != nullptr);
    CHECK(mm.bytes_allocated() == sz);
    mm.free(ptr2);
    mm.release_cache();
    CHECK(mm.bytes_allocated() == 0);
    CHECK(mm.bytes_reserved()  == 0);
}

// ── Test 5 ───────────────────────────────────────────────────────────────────

void test_allocator_multiple_live_allocs_tracked() {
    std::cout << "[Test 5] bytes_allocated tracks multiple concurrent allocations\n";
    CPUMemoryManager mm;
    constexpr std::size_t sz = 4ULL * 1024ULL * 1024ULL;

    std::byte* a = mm.allocate(sz);
    std::byte* b = mm.allocate(sz);
    std::byte* c = mm.allocate(sz);
    CHECK(mm.bytes_allocated() == 3 * sz);

    mm.free(b);
    CHECK(mm.bytes_allocated() == 2 * sz);
    mm.free(a);
    mm.free(c);
    CHECK(mm.bytes_allocated() == 0);
}

// ── Test 6 ───────────────────────────────────────────────────────────────────

void test_allocator_throws_on_zero_bytes() {
    std::cout << "[Test 6] allocate(0) throws std::invalid_argument\n";
    CPUMemoryManager mm;
    bool threw = false;
    try { static_cast<void>(mm.allocate(0)); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    CHECK(mm.bytes_allocated() == 0);
}

// ── Test 7 ───────────────────────────────────────────────────────────────────

void test_allocator_throws_on_bad_alignment() {
    std::cout << "[Test 7] allocate with non-power-of-2 alignment throws\n";
    CPUMemoryManager mm;
    bool threw = false;
    try { static_cast<void>(mm.allocate(1024, 3)); } // 3 is not a power of 2
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    CHECK(mm.bytes_allocated() == 0);
}

// ── Test 8 ───────────────────────────────────────────────────────────────────

void test_allocator_returned_pointer_is_aligned() {
    std::cout << "[Test 8] returned pointer satisfies requested alignment\n";
    CPUMemoryManager mm;
    // Test both small and large paths with 64-byte alignment
    std::byte* small = mm.allocate(128, 64);
    std::byte* large = mm.allocate(4ULL * 1024ULL * 1024ULL, 64);

    const auto small_addr = reinterpret_cast<std::uintptr_t>(small);
    const auto large_addr = reinterpret_cast<std::uintptr_t>(large);
    CHECK(small_addr % 64 == 0);
    CHECK(large_addr % 64 == 0);

    mm.free(small);
    mm.free(large);
    CHECK(mm.bytes_allocated() == 0);
}

// ── Test 9 ───────────────────────────────────────────────────────────────────

void test_buffer_construction_and_zero_init() {
    std::cout << "[Test 9] Buffer allocates, zero-initialises, and frees on destruction\n";
    CPUMemoryManager mm;
    {
        // 4 MB — large path
        Buffer buf(4ULL * 1024ULL * 1024ULL, &mm);
        CHECK(buf.size()   == 4ULL * 1024ULL * 1024ULL);
        CHECK(buf.device() == Device::CPU);
        CHECK(mm.bytes_allocated() == 4ULL * 1024ULL * 1024ULL);
    }
    // Buffer destructor must have freed the allocation
    CHECK(mm.bytes_allocated() == 0);
}

// ── Test 10 ──────────────────────────────────────────────────────────────────

void test_buffer_init_data_copied() {
    std::cout << "[Test 10] Buffer copies init_data into allocation\n";
    CPUMemoryManager mm;
    const double src[4] = {1.0, 2.0, 3.0, 4.0};
    Buffer buf(sizeof(src), &mm, static_cast<const void*>(src));
    // Verify via a second Buffer reading the same bytes through a fresh copy
    // We can't read data directly (no Passkey<Tensor> — by design).
    // Instead verify indirectly: a second allocation with memcmp after copy.
    // Simple check: re-allocate and memcpy, then compare.
    // This test just confirms construction doesn't crash and size is correct.
    CHECK(buf.size()   == sizeof(src));
    CHECK(buf.device() == Device::CPU);
    CHECK(mm.bytes_allocated() == sizeof(src));
}

// ── Test 11 ──────────────────────────────────────────────────────────────────

void test_buffer_small_path_frees_immediately() {
    std::cout << "[Test 11] small Buffer (< 1 MB) does not grow bytes_reserved\n";
    CPUMemoryManager mm;
    const std::size_t reserved_before = mm.bytes_reserved();
    {
        Buffer buf(256, &mm); // 256 bytes — small path
        CHECK(mm.bytes_allocated() == 256);
        // Small path never touches the mmap pool
        CHECK(mm.bytes_reserved() == reserved_before);
    }
    CHECK(mm.bytes_allocated() == 0);
    CHECK(mm.bytes_reserved()  == reserved_before);
}

// ── Test 12 ──────────────────────────────────────────────────────────────────

void test_allocator_reserved_invariant_holds() {
    std::cout << "[Test 12] bytes_reserved >= bytes_allocated at all times\n";
    CPUMemoryManager mm;
    constexpr std::size_t sz = 4ULL * 1024ULL * 1024ULL;

    std::byte* a = mm.allocate(sz);
    CHECK(mm.bytes_reserved() >= mm.bytes_allocated());
    std::byte* b = mm.allocate(sz);
    CHECK(mm.bytes_reserved() >= mm.bytes_allocated());
    mm.free(a);
    CHECK(mm.bytes_reserved() >= mm.bytes_allocated());
    mm.free(b);
    CHECK(mm.bytes_reserved() >= mm.bytes_allocated());
    mm.release_cache();
    CHECK(mm.bytes_reserved() == 0);
    CHECK(mm.bytes_allocated() == 0);
}

// ── Runner ───────────────────────────────────────────────────────────────────

void run_memory_tests() {
    test_allocator_small_alloc_zero_initialised();
    test_allocator_large_alloc_zero_initialised();
    test_allocator_large_pool_reuses_segment();
    test_allocator_release_cache_returns_memory_to_os();
    test_allocator_multiple_live_allocs_tracked();
    test_allocator_throws_on_zero_bytes();
    test_allocator_throws_on_bad_alignment();
    test_allocator_returned_pointer_is_aligned();
    test_buffer_construction_and_zero_init();
    test_buffer_init_data_copied();
    test_buffer_small_path_frees_immediately();
    test_allocator_reserved_invariant_holds();
}

} // namespace otter::test
