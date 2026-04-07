#include "test_utils.h"

#include <cstddef>
#include <memory>
#include <stdexcept>

#include "otter/tensor.h"
#include "otter/kernel/kernel_engine.h"
#include "otter/kernel/backend.h"
#include "memory/cpu_memory_manager.h"

namespace otter::test {

// ── Stub infrastructure ───────────────────────────────────────────────────────
//
// g_dispatcher_live tracks how many stub Dispatcher objects are currently
// alive. Every stub ctor increments; every stub dtor decrements.
// g_last_called_id records which dispatcher's call() was most recently invoked.
// Resetting both before each assertion sequence ensures independence.

inline int g_dispatcher_live  = 0;
inline int g_last_called_id   = -1;

struct StubBinaryDispatcher : KernelEngine::BinaryDispatcher {
    int id;
    explicit StubBinaryDispatcher(int i) : id(i) { ++g_dispatcher_live; }
    ~StubBinaryDispatcher() override              { --g_dispatcher_live; }
    void call(const Tensor&, const Tensor&, Tensor&) const override {
        g_last_called_id = id;
    }
};

struct StubUnaryDispatcher : KernelEngine::UnaryDispatcher {
    int id;
    explicit StubUnaryDispatcher(int i) : id(i) { ++g_dispatcher_live; }
    ~StubUnaryDispatcher() override              { --g_dispatcher_live; }
    void call(const Tensor&, Tensor&) const override { g_last_called_id = id; }
};

struct StubFillDispatcher : KernelEngine::FillDispatcher {
    int id;
    explicit StubFillDispatcher(int i) : id(i) { ++g_dispatcher_live; }
    ~StubFillDispatcher() override              { --g_dispatcher_live; }
    void call(Tensor&, double) const override   { g_last_called_id = id; }
};

struct StubReduceDispatcher : KernelEngine::ReduceDispatcher {
    int id;
    explicit StubReduceDispatcher(int i) : id(i) { ++g_dispatcher_live; }
    ~StubReduceDispatcher() override              { --g_dispatcher_live; }
    void call(const Tensor&, Tensor&) const override { g_last_called_id = id; }
};

struct StubMatMulDispatcher : KernelEngine::MatMulDispatcher {
    int id;
    explicit StubMatMulDispatcher(int i) : id(i) { ++g_dispatcher_live; }
    ~StubMatMulDispatcher() override              { --g_dispatcher_live; }
    void call(const Tensor&, const Tensor&, Tensor&) const override {
        g_last_called_id = id;
    }
};

struct StubCopyDispatcher : KernelEngine::CopyDispatcher {
    int id;
    explicit StubCopyDispatcher(int i) : id(i) { ++g_dispatcher_live; }
    ~StubCopyDispatcher() override              { --g_dispatcher_live; }
    void call(const Tensor&, Tensor&) const override { g_last_called_id = id; }
};

struct StubElementReadDispatcher : KernelEngine::ElementReadDispatcher {
    int    id;
    double return_value;
    explicit StubElementReadDispatcher(int i, double v = 42.0)
        : id(i), return_value(v) { ++g_dispatcher_live; }
    ~StubElementReadDispatcher() override { --g_dispatcher_live; }
    double call(const Tensor&, std::size_t) const override {
        g_last_called_id = id;
        return return_value;
    }
};

// Minimal concrete KernelEngine: exposes protected register_* as public
// so tests can populate the registry without subclassing a full backend.
struct TestEngine : KernelEngine {
    using KernelEngine::register_binary;
    using KernelEngine::register_unary;
    using KernelEngine::register_fill;
    using KernelEngine::register_reduce_to;
    using KernelEngine::register_matmul;
    using KernelEngine::register_copy;
    using KernelEngine::register_element_read;
};

// ── KernelEngine tests ────────────────────────────────────────────────────────

// Test 1 — empty registry throws on every dispatch family
void test_engine_empty_registry_throws_on_all_families() {
    std::cout << "[Test 1] empty KernelEngine throws on all dispatch calls\n";
    TestEngine engine;
    Tensor a, b, out;

    bool threw;

    threw = false;
    try { engine.dispatch_binary(KernelType::Add, a, b, out); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);

    threw = false;
    try { engine.dispatch_unary(KernelType::Neg, a, out); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);

    threw = false;
    try { engine.dispatch_fill(out, 0.0); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);

    threw = false;
    try { engine.dispatch_reduce_to(a, out); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);

    threw = false;
    try { engine.dispatch_matmul(a, b, out); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);

    threw = false;
    try { engine.dispatch_copy(a, out); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);

    threw = false;
    try { static_cast<void>(engine.dispatch_element_read(a, 0)); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

// Test 2 — registered binary ops route to the correct dispatcher
void test_engine_binary_ops_route_to_correct_dispatcher() {
    std::cout << "[Test 2] dispatch_binary routes each KernelType to its dispatcher\n";
    const int before = g_dispatcher_live;
    {
        TestEngine engine;
        engine.register_binary(KernelType::Add, std::make_unique<StubBinaryDispatcher>(1));
        engine.register_binary(KernelType::Mul, std::make_unique<StubBinaryDispatcher>(2));
        CHECK(g_dispatcher_live == before + 2);

        Tensor a, b, out;

        g_last_called_id = -1;
        engine.dispatch_binary(KernelType::Add, a, b, out);
        CHECK(g_last_called_id == 1);  // Add dispatcher

        g_last_called_id = -1;
        engine.dispatch_binary(KernelType::Mul, a, b, out);
        CHECK(g_last_called_id == 2);  // Mul dispatcher
    }
    CHECK(g_dispatcher_live == before);  // both freed on engine destruction
}

// Test 3 — unregistered op throws even when other ops are registered
void test_engine_unregistered_op_throws_with_others_present() {
    std::cout << "[Test 3] unregistered op throws even when other ops are present\n";
    const int before = g_dispatcher_live;
    {
        TestEngine engine;
        engine.register_binary(KernelType::Add, std::make_unique<StubBinaryDispatcher>(1));

        Tensor a, b, out;

        // Mul is not registered — must throw
        bool threw = false;
        try { engine.dispatch_binary(KernelType::Mul, a, b, out); }
        catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);

        // Add is registered — must succeed
        g_last_called_id = -1;
        engine.dispatch_binary(KernelType::Add, a, b, out);
        CHECK(g_last_called_id == 1);
    }
    CHECK(g_dispatcher_live == before);
}

// Test 4 — re-registering an op replaces the old dispatcher and frees it
void test_engine_reregistration_frees_old_and_replaces() {
    std::cout << "[Test 4] re-registering an op frees the old dispatcher and routes to the new one\n";
    const int before = g_dispatcher_live;
    {
        TestEngine engine;

        // Register first dispatcher (id=1)
        engine.register_binary(KernelType::Add, std::make_unique<StubBinaryDispatcher>(1));
        CHECK(g_dispatcher_live == before + 1);

        // Replace with second dispatcher (id=2) — first must be freed immediately
        engine.register_binary(KernelType::Add, std::make_unique<StubBinaryDispatcher>(2));
        CHECK(g_dispatcher_live == before + 1);  // still 1 alive, not 2

        Tensor a, b, out;
        g_last_called_id = -1;
        engine.dispatch_binary(KernelType::Add, a, b, out);
        CHECK(g_last_called_id == 2);  // new dispatcher
    }
    CHECK(g_dispatcher_live == before);  // second dispatcher freed on engine destruction
}

// Test 5 — all family dispatchers route correctly
void test_engine_all_families_route_correctly() {
    std::cout << "[Test 5] all dispatch families route to their registered implementations\n";
    const int before = g_dispatcher_live;
    {
        TestEngine engine;
        engine.register_unary(KernelType::Neg,   std::make_unique<StubUnaryDispatcher>(10));
        engine.register_fill(                    std::make_unique<StubFillDispatcher>(20));
        engine.register_reduce_to(               std::make_unique<StubReduceDispatcher>(30));
        engine.register_matmul(                  std::make_unique<StubMatMulDispatcher>(40));
        engine.register_copy(                    std::make_unique<StubCopyDispatcher>(50));
        engine.register_element_read(            std::make_unique<StubElementReadDispatcher>(60));
        CHECK(g_dispatcher_live == before + 6);

        Tensor a, b, out;

        g_last_called_id = -1;
        engine.dispatch_unary(KernelType::Neg, a, out);
        CHECK(g_last_called_id == 10);

        g_last_called_id = -1;
        engine.dispatch_fill(out, 1.0);
        CHECK(g_last_called_id == 20);

        g_last_called_id = -1;
        engine.dispatch_reduce_to(a, out);
        CHECK(g_last_called_id == 30);

        g_last_called_id = -1;
        engine.dispatch_matmul(a, b, out);
        CHECK(g_last_called_id == 40);

        g_last_called_id = -1;
        engine.dispatch_copy(a, out);
        CHECK(g_last_called_id == 50);

        g_last_called_id = -1;
        static_cast<void>(engine.dispatch_element_read(a, 0));
        CHECK(g_last_called_id == 60);
    }
    CHECK(g_dispatcher_live == before);
}

// Test 6 — dispatch_element_read return value passes through correctly
void test_engine_element_read_return_value_passes_through() {
    std::cout << "[Test 6] dispatch_element_read return value is forwarded from dispatcher\n";
    const int before = g_dispatcher_live;
    {
        TestEngine engine;
        // dispatcher returns 3.14
        engine.register_element_read(
            std::make_unique<StubElementReadDispatcher>(1, 3.14));

        Tensor t;
        const double result = engine.dispatch_element_read(t, 5);
        // value 3.14 must reach the caller unchanged
        CHECK_NEAR(result, 3.14, 1e-12);
    }
    CHECK(g_dispatcher_live == before);
}

// Test 7 — re-registering a family dispatcher frees the old one
void test_engine_family_reregistration_frees_old() {
    std::cout << "[Test 7] re-registering a family dispatcher frees the old and routes to the new\n";
    const int before = g_dispatcher_live;
    {
        TestEngine engine;
        engine.register_fill(std::make_unique<StubFillDispatcher>(1));
        CHECK(g_dispatcher_live == before + 1);

        engine.register_fill(std::make_unique<StubFillDispatcher>(2));
        CHECK(g_dispatcher_live == before + 1);  // first freed, second alive

        Tensor t;
        g_last_called_id = -1;
        engine.dispatch_fill(t, 0.0);
        CHECK(g_last_called_id == 2);
    }
    CHECK(g_dispatcher_live == before);
}

// Test 8 — KernelEngine destruction frees all registered dispatchers
void test_engine_destruction_frees_all_dispatchers() {
    std::cout << "[Test 8] KernelEngine destruction frees all registered dispatchers\n";
    const int before = g_dispatcher_live;
    {
        TestEngine engine;
        engine.register_binary(KernelType::Add,  std::make_unique<StubBinaryDispatcher>(1));
        engine.register_binary(KernelType::Mul,  std::make_unique<StubBinaryDispatcher>(2));
        engine.register_unary(KernelType::Neg,   std::make_unique<StubUnaryDispatcher>(3));
        engine.register_fill(                    std::make_unique<StubFillDispatcher>(4));
        engine.register_reduce_to(               std::make_unique<StubReduceDispatcher>(5));
        engine.register_matmul(                  std::make_unique<StubMatMulDispatcher>(6));
        engine.register_copy(                    std::make_unique<StubCopyDispatcher>(7));
        engine.register_element_read(            std::make_unique<StubElementReadDispatcher>(8));
        CHECK(g_dispatcher_live == before + 8);
    }
    // All 8 dispatchers freed when engine went out of scope
    CHECK(g_dispatcher_live == before);
}

// ── Backend tests ─────────────────────────────────────────────────────────────

// Test 9 — Backend::device() matches the MemoryManager's device
void test_backend_device_matches_memory_manager() {
    std::cout << "[Test 9] Backend::device() matches the MemoryManager's device\n";
    Backend backend(
        std::make_unique<CPUMemoryManager>(),
        std::make_unique<TestEngine>()
    );
    CHECK(backend.device() == Device::CPU);
}

// Test 10 — Backend accessors return the exact MM and KE passed at construction
void test_backend_accessors_return_correct_pointers() {
    std::cout << "[Test 10] Backend accessors return the same objects passed at construction\n";
    auto mm = std::make_unique<CPUMemoryManager>();
    auto ke = std::make_unique<TestEngine>();

    // Capture raw pointers before move — used as identity witnesses
    MemoryManager* mm_ptr = mm.get();
    KernelEngine*  ke_ptr = ke.get();

    Backend backend(std::move(mm), std::move(ke));

    CHECK(backend.memory_manager() != nullptr);
    CHECK(backend.kernel_engine()  != nullptr);
    CHECK(backend.memory_manager() == mm_ptr);
    CHECK(backend.kernel_engine()  == ke_ptr);
}

// Test 11 — const Backend accessors compile and return same pointers
void test_backend_const_accessors_work() {
    std::cout << "[Test 11] const Backend accessors return non-null and match device\n";
    auto mm = std::make_unique<CPUMemoryManager>();
    auto ke = std::make_unique<TestEngine>();
    MemoryManager* mm_ptr = mm.get();
    KernelEngine*  ke_ptr = ke.get();

    Backend backend(std::move(mm), std::move(ke));
    const Backend& cb = backend;

    CHECK(cb.memory_manager() != nullptr);
    CHECK(cb.kernel_engine()  != nullptr);
    CHECK(cb.memory_manager() == mm_ptr);
    CHECK(cb.kernel_engine()  == ke_ptr);
    CHECK(cb.device()         == Device::CPU);
}

// Test 12 — Backend destruction frees KernelEngine and all its dispatchers
void test_backend_destruction_frees_kernel_engine() {
    std::cout << "[Test 12] Backend destruction frees KernelEngine and all its dispatchers\n";
    const int before = g_dispatcher_live;
    {
        auto ke = std::make_unique<TestEngine>();
        ke->register_binary(KernelType::Add, std::make_unique<StubBinaryDispatcher>(1));
        ke->register_fill(                   std::make_unique<StubFillDispatcher>(2));
        ke->register_element_read(           std::make_unique<StubElementReadDispatcher>(3));
        CHECK(g_dispatcher_live == before + 3);

        {
            Backend backend(
                std::make_unique<CPUMemoryManager>(),
                std::move(ke)
            );
            CHECK(g_dispatcher_live == before + 3);  // still alive inside backend
        }
        // backend destructed → unique_ptr<KernelEngine> → ~TestEngine → ~StubXxxDispatcher
        CHECK(g_dispatcher_live == before);
    }
}

// Test 13 — allocations through Backend's MemoryManager are tracked and freed
void test_backend_memory_manager_allocation_tracking() {
    std::cout << "[Test 13] allocations via Backend's MemoryManager are tracked and freed\n";
    Backend backend(
        std::make_unique<CPUMemoryManager>(),
        std::make_unique<TestEngine>()
    );

    MemoryManager* mm = backend.memory_manager();

    // Initial state: nothing allocated
    CHECK(mm->bytes_allocated() == 0);

    // Allocate two blocks
    constexpr std::size_t sz_a = 256;
    constexpr std::size_t sz_b = 512;
    std::byte* a = mm->allocate(sz_a);
    std::byte* b = mm->allocate(sz_b);
    CHECK(mm->bytes_allocated() == sz_a + sz_b);

    // Free one — tracking must update immediately
    mm->free(a);
    CHECK(mm->bytes_allocated() == sz_b);

    mm->free(b);
    CHECK(mm->bytes_allocated() == 0);
}

// Test 14 — large allocation through Backend, pool reserved, cleared on release
void test_backend_large_alloc_pool_and_release() {
    std::cout << "[Test 14] large alloc through Backend pools segment; release_cache clears it\n";
    Backend backend(
        std::make_unique<CPUMemoryManager>(),
        std::make_unique<TestEngine>()
    );

    MemoryManager* mm = backend.memory_manager();
    constexpr std::size_t sz = 4ULL * 1024ULL * 1024ULL;  // 4 MB — large path

    std::byte* ptr = mm->allocate(sz);
    CHECK(mm->bytes_allocated() == sz);
    CHECK(mm->bytes_reserved()  >= sz);

    mm->free(ptr);
    CHECK(mm->bytes_allocated() == 0);
    CHECK(mm->bytes_reserved()  >= sz);  // segment cached, not yet munmap'd

    mm->release_cache();
    CHECK(mm->bytes_reserved()  == 0);
    CHECK(mm->bytes_allocated() == 0);
}

// Test 15 — KernelEngine passed to Backend is still functional after move
void test_backend_engine_is_functional_after_move() {
    std::cout << "[Test 15] KernelEngine inside Backend dispatches correctly after move\n";
    const int before = g_dispatcher_live;
    {
        auto ke = std::make_unique<TestEngine>();
        ke->register_binary(KernelType::Add, std::make_unique<StubBinaryDispatcher>(99));

        Backend backend(std::make_unique<CPUMemoryManager>(), std::move(ke));

        Tensor a, b, out;
        g_last_called_id = -1;
        backend.kernel_engine()->dispatch_binary(KernelType::Add, a, b, out);
        CHECK(g_last_called_id == 99);
    }
    CHECK(g_dispatcher_live == before);
}

// Test 16 — Backend with unreleased pool destructs without crash or leak
//
// This is the "end of program" scenario: allocate a large block through the
// backend, free it (block goes back to MM pool, reserved > 0), then let the
// backend go out of scope WITHOUT calling release_cache. The CPUMemoryManager
// destructor must munmap the cached segment. If it doesn't, the test either
// crashes (ASan/UBSan) or the OS silently reclaims without our destructor
// running properly.
void test_backend_destruction_with_unreleased_pool() {
    std::cout << "[Test 16] Backend destructs cleanly with unreleased pool segments\n";
    constexpr std::size_t sz = 4ULL * 1024ULL * 1024ULL;  // 4 MB — large path

    {
        Backend backend(
            std::make_unique<CPUMemoryManager>(),
            std::make_unique<TestEngine>()
        );

        MemoryManager* mm = backend.memory_manager();

        std::byte* ptr = mm->allocate(sz);
        CHECK(mm->bytes_allocated() == sz);

        mm->free(ptr);
        // Pool segment is cached (reserved > 0) but not released.
        // We intentionally do NOT call mm->release_cache() here.
        CHECK(mm->bytes_allocated() == 0);
        CHECK(mm->bytes_reserved()  >= sz);

        // Backend goes out of scope here. ~CPUMemoryManager() must munmap the
        // cached segment. ASan/UBSan in debug build will detect any bad access.
    }
    // If we reach here without a crash, the destructor ran cleanly. ✓
    CHECK(true);
}

// ── Runner ────────────────────────────────────────────────────────────────────

void run_kernel_tests() {
    test_engine_empty_registry_throws_on_all_families();
    test_engine_binary_ops_route_to_correct_dispatcher();
    test_engine_unregistered_op_throws_with_others_present();
    test_engine_reregistration_frees_old_and_replaces();
    test_engine_all_families_route_correctly();
    test_engine_element_read_return_value_passes_through();
    test_engine_family_reregistration_frees_old();
    test_engine_destruction_frees_all_dispatchers();
    test_backend_device_matches_memory_manager();
    test_backend_accessors_return_correct_pointers();
    test_backend_const_accessors_work();
    test_backend_destruction_frees_kernel_engine();
    test_backend_memory_manager_allocation_tracking();
    test_backend_large_alloc_pool_and_release();
    test_backend_engine_is_functional_after_move();
    test_backend_destruction_with_unreleased_pool();
}

} // namespace otter::test
