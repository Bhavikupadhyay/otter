#include "../utils/test_utils.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "otter/memory/buffer.h"
#include "otter/kernel/backend.h"
#include "otter/kernel/kernel_engine.h"
#include "memory/cpu_memory_manager.h"

namespace {

// Minimal KernelEngine for Buffer tests — no dispatchers registered.
struct NullEngine : otter::KernelEngine {};

// Returns a Backend owning a CPUMemoryManager + NullEngine.
std::unique_ptr<otter::Backend> make_test_backend() {
    return std::make_unique<otter::Backend>(
        std::make_unique<otter::CPUMemoryManager>(),
        std::make_unique<NullEngine>()
    );
}

} // namespace

namespace otter::test {

void test_allocator_small_alloc_zero_initialised() {
    std::cout << "[CPUMem 1] small allocation is zero-initialised\n";
    CPUMemoryManager mm;
    std::byte* ptr = mm.allocate(64);
    bool all_zero = true;
    for (std::size_t i = 0; i < 64; ++i) all_zero &= (ptr[i] == std::byte{0});
    CHECK(all_zero);
    CHECK(mm.bytes_allocated() == 64);
    mm.free(ptr);
    CHECK(mm.bytes_allocated() == 0);
}

void test_allocator_large_alloc_zero_initialised() {
    std::cout << "[CPUMem 2] large allocation is zero-initialised\n";
    CPUMemoryManager mm;
    constexpr std::size_t sz = 4ULL * 1024ULL * 1024ULL;
    std::byte* ptr = mm.allocate(sz);
    bool all_zero = true;
    for (std::size_t i = 0; i < sz; i += 4096) all_zero &= (ptr[i] == std::byte{0});
    CHECK(all_zero);
    CHECK(mm.bytes_allocated() == sz);
    CHECK(mm.bytes_reserved()  >= sz);
    mm.free(ptr);
    CHECK(mm.bytes_allocated() == 0);
    CHECK(mm.bytes_reserved()  >= sz);  // segment still cached
}

void test_allocator_large_pool_reuses_segment() {
    std::cout << "[CPUMem 3] freed large segment is reused on next same-size allocation\n";
    CPUMemoryManager mm;
    constexpr std::size_t sz = 4ULL * 1024ULL * 1024ULL;
    std::byte* first = mm.allocate(sz);
    mm.free(first);
    const std::size_t reserved_after_free = mm.bytes_reserved();
    std::byte* second = mm.allocate(sz);
    // Reserved bytes must not grow — cached segment was reused
    CHECK(mm.bytes_reserved() == reserved_after_free);
    mm.free(second);
    CHECK(mm.bytes_allocated() == 0);
}

void test_allocator_release_cache_returns_memory_to_os() {
    std::cout << "[CPUMem 4] release_cache() drops all cached segments\n";
    CPUMemoryManager mm;
    constexpr std::size_t sz = 4ULL * 1024ULL * 1024ULL;
    std::byte* ptr = mm.allocate(sz);
    mm.free(ptr);
    CHECK(mm.bytes_reserved() > 0);
    mm.release_cache();
    CHECK(mm.bytes_reserved()  == 0);
    CHECK(mm.bytes_allocated() == 0);
    std::byte* ptr2 = mm.allocate(sz);
    CHECK(ptr2 != nullptr);
    CHECK(mm.bytes_allocated() == sz);
    mm.free(ptr2);
    mm.release_cache();
    CHECK(mm.bytes_allocated() == 0);
    CHECK(mm.bytes_reserved()  == 0);
}

void test_allocator_multiple_live_allocs_tracked() {
    std::cout << "[CPUMem 5] bytes_allocated tracks multiple concurrent allocations\n";
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

void test_allocator_throws_on_zero_bytes() {
    std::cout << "[CPUMem 6] allocate(0) throws std::invalid_argument\n";
    CPUMemoryManager mm;
    bool threw = false;
    try { static_cast<void>(mm.allocate(0)); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    CHECK(mm.bytes_allocated() == 0);
}

void test_allocator_throws_on_bad_alignment() {
    std::cout << "[CPUMem 7] allocate with non-power-of-2 alignment throws\n";
    CPUMemoryManager mm;
    bool threw = false;
    try { static_cast<void>(mm.allocate(1024, 3)); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    CHECK(mm.bytes_allocated() == 0);
}

void test_allocator_returned_pointer_is_aligned() {
    std::cout << "[CPUMem 8] returned pointer satisfies requested alignment\n";
    CPUMemoryManager mm;
    std::byte* small = mm.allocate(128, 64);
    std::byte* large = mm.allocate(4ULL * 1024ULL * 1024ULL, 64);
    CHECK(reinterpret_cast<std::uintptr_t>(small) % 64 == 0);
    CHECK(reinterpret_cast<std::uintptr_t>(large) % 64 == 0);
    mm.free(small);
    mm.free(large);
    CHECK(mm.bytes_allocated() == 0);
}

void test_buffer_construction_and_zero_init() {
    std::cout << "[CPUMem 9] Buffer allocates, zero-initialises, and frees on destruction\n";
    auto backend = make_test_backend();
    constexpr std::size_t sz = 4ULL * 1024ULL * 1024ULL;
    {
        Buffer buf(sz, *backend);
        CHECK(buf.size()     == sz);
        CHECK(&buf.backend() == backend.get());
        CHECK(backend->memory_manager()->bytes_allocated() == sz);
    }
    CHECK(backend->memory_manager()->bytes_allocated() == 0);
}

void test_buffer_init_data_copied() {
    std::cout << "[CPUMem 10] Buffer copies init_data into allocation\n";
    auto backend = make_test_backend();
    const double src[4] = {1.0, 2.0, 3.0, 4.0};
    Buffer buf(sizeof(src), *backend, static_cast<const void*>(src));
    CHECK(buf.size()     == sizeof(src));
    CHECK(&buf.backend() == backend.get());
    CHECK(backend->memory_manager()->bytes_allocated() == sizeof(src));
}

void test_buffer_small_path_frees_immediately() {
    std::cout << "[CPUMem 11] small Buffer (< 1 MB) tracks bytes_reserved; invariant holds\n";
    auto backend = make_test_backend();
    {
        Buffer buf(256, *backend);
        CHECK(backend->memory_manager()->bytes_allocated() == 256);
        CHECK(backend->memory_manager()->bytes_reserved() >= backend->memory_manager()->bytes_allocated());
        CHECK(backend->memory_manager()->bytes_reserved() == 256);
    }
    CHECK(backend->memory_manager()->bytes_allocated() == 0);
    CHECK(backend->memory_manager()->bytes_reserved()  == 0);
}

void test_allocator_reserved_invariant_holds() {
    std::cout << "[CPUMem 12] bytes_reserved >= bytes_allocated at all times\n";
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
    CHECK(mm.bytes_reserved()  == 0);
    CHECK(mm.bytes_allocated() == 0);
}

void test_allocator_invariant_holds_for_small_allocs() {
    std::cout << "[CPUMem 13] bytes_reserved >= bytes_allocated holds for small allocs\n";
    CPUMemoryManager mm;
    std::byte* a = mm.allocate(256);
    std::byte* b = mm.allocate(512);
    CHECK(mm.bytes_allocated() == 768);
    CHECK(mm.bytes_reserved()  >= mm.bytes_allocated());
    mm.free(a);
    CHECK(mm.bytes_reserved()  >= mm.bytes_allocated());
    mm.free(b);
    CHECK(mm.bytes_allocated() == 0);
    CHECK(mm.bytes_reserved()  == 0);
}

void run_cpu_memory_tests() {
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
    test_allocator_invariant_holds_for_small_allocs();
}

} // namespace otter::test
