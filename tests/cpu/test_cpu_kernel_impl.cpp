#include "../utils/test_utils.h"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "otter/tensor.h"
#include "otter/backends/cpu.h"
#include "otter/kernel/backend.h"
#include "otter/kernel/kernel_engine.h"
#include "otter/optim/sgd.h"
#include "cpu/memory/cpu_memory_manager.h"

// CPU kernel implementation tests: direct dispatch_binary/unary/copy/matmul,
// kernel registry stubs, SGD/dispatch_scale/dispatch_axpy, contiguous(), MLP loop.
// These call into CPUKernelEngine dispatch methods directly — CPU-specific by nature.

namespace otter::test {

// ── Stub infrastructure (kernel registry tests) ───────────────────────────────

namespace {

inline int g_dispatcher_live  = 0;
inline int g_last_called_id   = -1;

struct StubBinaryDispatcher : KernelEngine::BinaryDispatcher {
    int id;
    explicit StubBinaryDispatcher(int i) : id(i) { ++g_dispatcher_live; }
    ~StubBinaryDispatcher() override              { --g_dispatcher_live; }
    void call(const Tensor&, const Tensor&, Tensor&) const override { g_last_called_id = id; }
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

struct StubReduceDispatcher : KernelEngine::UnaryDispatcher {
    int id;
    explicit StubReduceDispatcher(int i) : id(i) { ++g_dispatcher_live; }
    ~StubReduceDispatcher() override              { --g_dispatcher_live; }
    void call(const Tensor&, Tensor&) const override { g_last_called_id = id; }
};

struct StubMatMulDispatcher : KernelEngine::MatMulDispatcher {
    int id;
    explicit StubMatMulDispatcher(int i) : id(i) { ++g_dispatcher_live; }
    ~StubMatMulDispatcher() override              { --g_dispatcher_live; }
    void call(const Tensor&, const Tensor&, Tensor&) const override { g_last_called_id = id; }
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

struct TestEngine : KernelEngine {
    using KernelEngine::register_binary;
    using KernelEngine::register_unary;
    using KernelEngine::register_fill;
    using KernelEngine::register_matmul;
    using KernelEngine::register_copy;
    using KernelEngine::register_element_read;
};

} // namespace

// ── KernelEngine registry tests ───────────────────────────────────────────────

void test_engine_empty_registry_throws_on_all_families() {
    std::cout << "[KernelReg 1] empty KernelEngine throws on all dispatch calls\n";
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
    try { engine.dispatch_unary(KernelType::ReduceTo, a, out); }
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

void test_engine_binary_ops_route_to_correct_dispatcher() {
    std::cout << "[KernelReg 2] dispatch_binary routes each KernelType to its dispatcher\n";
    const int before = g_dispatcher_live;
    {
        TestEngine engine;
        engine.register_binary(KernelType::Add, std::make_unique<StubBinaryDispatcher>(1));
        engine.register_binary(KernelType::Mul, std::make_unique<StubBinaryDispatcher>(2));
        CHECK(g_dispatcher_live == before + 2);
        Tensor a, b, out;
        g_last_called_id = -1;
        engine.dispatch_binary(KernelType::Add, a, b, out);
        CHECK(g_last_called_id == 1);
        g_last_called_id = -1;
        engine.dispatch_binary(KernelType::Mul, a, b, out);
        CHECK(g_last_called_id == 2);
    }
    CHECK(g_dispatcher_live == before);
}

void test_engine_unregistered_op_throws_with_others_present() {
    std::cout << "[KernelReg 3] unregistered op throws even when other ops are present\n";
    const int before = g_dispatcher_live;
    {
        TestEngine engine;
        engine.register_binary(KernelType::Add, std::make_unique<StubBinaryDispatcher>(1));
        Tensor a, b, out;
        bool threw = false;
        try { engine.dispatch_binary(KernelType::Mul, a, b, out); }
        catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
        g_last_called_id = -1;
        engine.dispatch_binary(KernelType::Add, a, b, out);
        CHECK(g_last_called_id == 1);
    }
    CHECK(g_dispatcher_live == before);
}

void test_engine_reregistration_frees_old_and_replaces() {
    std::cout << "[KernelReg 4] re-registering frees old dispatcher and routes to new one\n";
    const int before = g_dispatcher_live;
    {
        TestEngine engine;
        engine.register_binary(KernelType::Add, std::make_unique<StubBinaryDispatcher>(1));
        CHECK(g_dispatcher_live == before + 1);
        engine.register_binary(KernelType::Add, std::make_unique<StubBinaryDispatcher>(2));
        CHECK(g_dispatcher_live == before + 1);  // first freed immediately
        Tensor a, b, out;
        g_last_called_id = -1;
        engine.dispatch_binary(KernelType::Add, a, b, out);
        CHECK(g_last_called_id == 2);
    }
    CHECK(g_dispatcher_live == before);
}

void test_engine_all_families_route_correctly() {
    std::cout << "[KernelReg 5] all dispatch families route to their registered implementations\n";
    const int before = g_dispatcher_live;
    {
        TestEngine engine;
        engine.register_unary(KernelType::Neg,      std::make_unique<StubUnaryDispatcher>(10));
        engine.register_fill(                        std::make_unique<StubFillDispatcher>(20));
        engine.register_unary(KernelType::ReduceTo,  std::make_unique<StubReduceDispatcher>(30));
        engine.register_matmul(                      std::make_unique<StubMatMulDispatcher>(40));
        engine.register_copy(                        std::make_unique<StubCopyDispatcher>(50));
        engine.register_element_read(                std::make_unique<StubElementReadDispatcher>(60));
        CHECK(g_dispatcher_live == before + 6);
        Tensor a, b, out;
        g_last_called_id = -1; engine.dispatch_unary(KernelType::Neg, a, out);    CHECK(g_last_called_id == 10);
        g_last_called_id = -1; engine.dispatch_fill(out, 1.0);                    CHECK(g_last_called_id == 20);
        g_last_called_id = -1; engine.dispatch_unary(KernelType::ReduceTo, a, out); CHECK(g_last_called_id == 30);
        g_last_called_id = -1; engine.dispatch_matmul(a, b, out);                 CHECK(g_last_called_id == 40);
        g_last_called_id = -1; engine.dispatch_copy(a, out);                      CHECK(g_last_called_id == 50);
        g_last_called_id = -1; static_cast<void>(engine.dispatch_element_read(a, 0)); CHECK(g_last_called_id == 60);
    }
    CHECK(g_dispatcher_live == before);
}

void test_engine_element_read_return_value_passes_through() {
    std::cout << "[KernelReg 6] dispatch_element_read return value is forwarded from dispatcher\n";
    const int before = g_dispatcher_live;
    {
        TestEngine engine;
        engine.register_element_read(std::make_unique<StubElementReadDispatcher>(1, 3.14));
        Tensor t;
        const double result = engine.dispatch_element_read(t, 5);
        CHECK_NEAR(result, 3.14, 1e-12);
    }
    CHECK(g_dispatcher_live == before);
}

void test_engine_family_reregistration_frees_old() {
    std::cout << "[KernelReg 7] re-registering a family dispatcher frees the old\n";
    const int before = g_dispatcher_live;
    {
        TestEngine engine;
        engine.register_fill(std::make_unique<StubFillDispatcher>(1));
        CHECK(g_dispatcher_live == before + 1);
        engine.register_fill(std::make_unique<StubFillDispatcher>(2));
        CHECK(g_dispatcher_live == before + 1);
        Tensor t;
        g_last_called_id = -1;
        engine.dispatch_fill(t, 0.0);
        CHECK(g_last_called_id == 2);
    }
    CHECK(g_dispatcher_live == before);
}

void test_engine_destruction_frees_all_dispatchers() {
    std::cout << "[KernelReg 8] KernelEngine destruction frees all registered dispatchers\n";
    const int before = g_dispatcher_live;
    {
        TestEngine engine;
        engine.register_binary(KernelType::Add,      std::make_unique<StubBinaryDispatcher>(1));
        engine.register_binary(KernelType::Mul,      std::make_unique<StubBinaryDispatcher>(2));
        engine.register_unary(KernelType::Neg,       std::make_unique<StubUnaryDispatcher>(3));
        engine.register_fill(                         std::make_unique<StubFillDispatcher>(4));
        engine.register_unary(KernelType::ReduceTo,  std::make_unique<StubReduceDispatcher>(5));
        engine.register_matmul(                       std::make_unique<StubMatMulDispatcher>(6));
        engine.register_copy(                         std::make_unique<StubCopyDispatcher>(7));
        engine.register_element_read(                 std::make_unique<StubElementReadDispatcher>(8));
        CHECK(g_dispatcher_live == before + 8);
    }
    CHECK(g_dispatcher_live == before);
}

// ── Backend tests ─────────────────────────────────────────────────────────────

void test_backend_device_matches_memory_manager() {
    std::cout << "[KernelReg 9] Backend::device() matches the MemoryManager's device\n";
    Backend backend(std::make_unique<CPUMemoryManager>(), std::make_unique<TestEngine>());
    CHECK(backend.device() == Device::CPU);
}

void test_backend_accessors_return_correct_pointers() {
    std::cout << "[KernelReg 10] Backend accessors return the same objects passed at construction\n";
    auto mm = std::make_unique<CPUMemoryManager>();
    auto ke = std::make_unique<TestEngine>();
    MemoryManager* mm_ptr = mm.get();
    KernelEngine*  ke_ptr = ke.get();
    Backend backend(std::move(mm), std::move(ke));
    CHECK(backend.memory_manager() == mm_ptr);
    CHECK(backend.kernel_engine()  == ke_ptr);
}

void test_backend_const_accessors_work() {
    std::cout << "[KernelReg 11] const Backend accessors return non-null and match device\n";
    auto mm = std::make_unique<CPUMemoryManager>();
    auto ke = std::make_unique<TestEngine>();
    MemoryManager* mm_ptr = mm.get();
    KernelEngine*  ke_ptr = ke.get();
    Backend backend(std::move(mm), std::move(ke));
    const Backend& cb = backend;
    CHECK(cb.memory_manager() == mm_ptr);
    CHECK(cb.kernel_engine()  == ke_ptr);
    CHECK(cb.device()         == Device::CPU);
}

void test_backend_destruction_frees_kernel_engine() {
    std::cout << "[KernelReg 12] Backend destruction frees KernelEngine and all its dispatchers\n";
    const int before = g_dispatcher_live;
    {
        auto ke = std::make_unique<TestEngine>();
        ke->register_binary(KernelType::Add, std::make_unique<StubBinaryDispatcher>(1));
        ke->register_fill(                   std::make_unique<StubFillDispatcher>(2));
        ke->register_element_read(           std::make_unique<StubElementReadDispatcher>(3));
        CHECK(g_dispatcher_live == before + 3);
        {
            Backend backend(std::make_unique<CPUMemoryManager>(), std::move(ke));
            CHECK(g_dispatcher_live == before + 3);
        }
        CHECK(g_dispatcher_live == before);
    }
}

void test_backend_memory_manager_allocation_tracking() {
    std::cout << "[KernelReg 13] allocations via Backend's MemoryManager are tracked and freed\n";
    Backend backend(std::make_unique<CPUMemoryManager>(), std::make_unique<TestEngine>());
    MemoryManager* mm = backend.memory_manager();
    CHECK(mm->bytes_allocated() == 0);
    std::byte* a = mm->allocate(256);
    std::byte* b = mm->allocate(512);
    CHECK(mm->bytes_allocated() == 256 + 512);
    mm->free(a);
    CHECK(mm->bytes_allocated() == 512);
    mm->free(b);
    CHECK(mm->bytes_allocated() == 0);
}

void test_backend_large_alloc_pool_and_release() {
    std::cout << "[KernelReg 14] large alloc pools segment; release_cache clears it\n";
    Backend backend(std::make_unique<CPUMemoryManager>(), std::make_unique<TestEngine>());
    MemoryManager* mm = backend.memory_manager();
    constexpr std::size_t sz = 4ULL * 1024ULL * 1024ULL;
    std::byte* ptr = mm->allocate(sz);
    CHECK(mm->bytes_allocated() == sz);
    CHECK(mm->bytes_reserved()  >= sz);
    mm->free(ptr);
    CHECK(mm->bytes_allocated() == 0);
    CHECK(mm->bytes_reserved()  >= sz);
    mm->release_cache();
    CHECK(mm->bytes_reserved()  == 0);
    CHECK(mm->bytes_allocated() == 0);
}

void test_backend_engine_is_functional_after_move() {
    std::cout << "[KernelReg 15] KernelEngine inside Backend dispatches correctly after move\n";
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

void test_backend_destruction_with_unreleased_pool() {
    std::cout << "[KernelReg 16] Backend destructs cleanly with unreleased pool segments\n";
    constexpr std::size_t sz = 4ULL * 1024ULL * 1024ULL;
    {
        Backend backend(std::make_unique<CPUMemoryManager>(), std::make_unique<TestEngine>());
        MemoryManager* mm = backend.memory_manager();
        std::byte* ptr = mm->allocate(sz);
        CHECK(mm->bytes_allocated() == sz);
        mm->free(ptr);
        CHECK(mm->bytes_allocated() == 0);
        CHECK(mm->bytes_reserved()  >= sz);
        // Intentionally do NOT call release_cache — destructor must handle it.
    }
    CHECK(true);
}

// ── CPUKernelEngine direct dispatch tests ─────────────────────────────────────

void test_fill_2d() {
    std::cout << "[CPUKernel 1] fill: {3,4} filled with 7.0\n";
    Backend& be = cpu_backend();
    Tensor t = Tensor::zeros({3, 4}, be);
    t.fill_(7.0);
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 4; ++c)
            CHECK_NEAR(t.at({r, c}), 7.0, 1e-12);
}

void test_fill_scalar() {
    std::cout << "[CPUKernel 2] fill: {1} with -3.5\n";
    Backend& be = cpu_backend();
    Tensor t = Tensor::zeros({1}, be);
    t.fill_(-3.5);
    CHECK_NEAR(t.at({0}), -3.5, 1e-12);
}

void test_add_contiguous() {
    std::cout << "[CPUKernel 3] add: {2,3} + {2,3} contiguous\n";
    Backend& be = cpu_backend();
    Tensor a   = Tensor::from_data<double>({1,2,3,4,5,6},       {2,3}, be);
    Tensor b   = Tensor::from_data<double>({10,20,30,40,50,60}, {2,3}, be);
    Tensor out = Tensor::zeros({2, 3}, be);
    be.kernel_engine()->dispatch_binary(KernelType::Add, a, b, out);
    CHECK_NEAR(out.at({0,0}), 11.0, 1e-12);
    CHECK_NEAR(out.at({0,1}), 22.0, 1e-12);
    CHECK_NEAR(out.at({0,2}), 33.0, 1e-12);
    CHECK_NEAR(out.at({1,0}), 44.0, 1e-12);
    CHECK_NEAR(out.at({1,1}), 55.0, 1e-12);
    CHECK_NEAR(out.at({1,2}), 66.0, 1e-12);
}

void test_add_broadcast_row() {
    std::cout << "[CPUKernel 4] add: broadcast {3} + {2,3} via stride-0 view\n";
    Backend& be = cpu_backend();
    Tensor row    = Tensor::from_data<double>({10, 20, 30}, {3}, be);
    Tensor row_bc = row.view({2, 3}, {0, 1});
    Tensor mat    = Tensor::from_data<double>({1, 2, 3, 4, 5, 6}, {2, 3}, be);
    Tensor out    = Tensor::zeros({2, 3}, be);
    be.kernel_engine()->dispatch_binary(KernelType::Add, row_bc, mat, out);
    CHECK_NEAR(out.at({0,0}), 11.0, 1e-12);
    CHECK_NEAR(out.at({1,2}), 36.0, 1e-12);
}

void test_mul_contiguous() {
    std::cout << "[CPUKernel 5] mul: {2,3} * {2,3}\n";
    Backend& be = cpu_backend();
    Tensor a   = Tensor::from_data<double>({1,2,3,4,5,6}, {2,3}, be);
    Tensor b   = Tensor::from_data<double>({2,3,4,5,6,7}, {2,3}, be);
    Tensor out = Tensor::zeros({2, 3}, be);
    be.kernel_engine()->dispatch_binary(KernelType::Mul, a, b, out);
    CHECK_NEAR(out.at({0,0}),  2.0, 1e-12);
    CHECK_NEAR(out.at({1,2}), 42.0, 1e-12);
}

void test_element_read_scalar() {
    std::cout << "[CPUKernel 6] element_read: {1} scalar\n";
    Backend& be = cpu_backend();
    Tensor t = Tensor::from_data<double>({42.0}, {1}, be);
    CHECK_NEAR(t.at({0}), 42.0, 1e-12);
}

void test_element_read_2d() {
    std::cout << "[CPUKernel 7] element_read: {2,3} at (1,2)\n";
    Backend& be = cpu_backend();
    Tensor t = Tensor::from_data<double>({10, 20, 30, 40, 50, 60}, {2, 3}, be);
    CHECK_NEAR(t.at({1, 2}), 60.0, 1e-12);
    CHECK_NEAR(t.at({0, 0}), 10.0, 1e-12);
}

void test_copy_contiguous() {
    std::cout << "[CPUKernel 8] copy: contiguous → contiguous\n";
    Backend& be = cpu_backend();
    Tensor src = Tensor::from_data<double>({1, 2, 3, 4, 5, 6}, {2, 3}, be);
    Tensor dst = Tensor::zeros({2, 3}, be);
    be.kernel_engine()->dispatch_copy(src, dst);
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            CHECK_NEAR(dst.at({r, c}), src.at({r, c}), 1e-12);
}

void test_copy_strided_view() {
    std::cout << "[CPUKernel 9] copy: non-contiguous view\n";
    Backend& be = cpu_backend();
    Tensor base = Tensor::from_data<double>({1, 2, 3, 4, 5, 6}, {2, 3}, be);
    Tensor view = base.view({3, 2}, {1, 3});
    Tensor dst  = Tensor::zeros({3, 2}, be);
    be.kernel_engine()->dispatch_copy(view, dst);
    CHECK_NEAR(dst.at({0,0}), 1.0, 1e-12);
    CHECK_NEAR(dst.at({0,1}), 4.0, 1e-12);
    CHECK_NEAR(dst.at({2,1}), 6.0, 1e-12);
}

void test_copy_stride0_broadcast() {
    std::cout << "[CPUKernel 10] copy: stride-0 broadcast view materialised correctly\n";
    Backend& be = cpu_backend();
    Tensor row = Tensor::from_data<double>({10, 20, 30}, {3}, be);
    Tensor bc  = row.view({4, 3}, {0, 1});
    Tensor dst = Tensor::zeros({4, 3}, be);
    be.kernel_engine()->dispatch_copy(bc, dst);
    for (std::size_t r = 0; r < 4; ++r) {
        CHECK_NEAR(dst.at({r, 0}), 10.0, 1e-12);
        CHECK_NEAR(dst.at({r, 2}), 30.0, 1e-12);
    }
}

void test_sum_2d() {
    std::cout << "[CPUKernel 11] sum: {3,4} → scalar == 78\n";
    Backend& be = cpu_backend();
    Tensor a   = Tensor::from_data<double>({1,2,3,4,5,6,7,8,9,10,11,12}, {3, 4}, be);
    Tensor out = Tensor::zeros({1}, be);
    be.kernel_engine()->dispatch_unary(KernelType::ReduceSum, a, out);
    CHECK_NEAR(out.at({0}), 78.0, 1e-12);
}

void test_sum_noncontiguous() {
    std::cout << "[CPUKernel 12] sum: non-contiguous — contiguous() first, result correct\n";
    Backend& be = cpu_backend();
    Tensor base = Tensor::from_data<double>({1,2,3,4,5,6}, {2,3}, be);
    Tensor view = base.view({3, 2}, {1, 3});
    Tensor cont = view.contiguous();
    Tensor out  = Tensor::zeros({1}, be);
    be.kernel_engine()->dispatch_unary(KernelType::ReduceSum, cont, out);
    CHECK_NEAR(out.at({0}), 21.0, 1e-12);
}

void test_reduce_to_rows() {
    std::cout << "[CPUKernel 13] reduce_to: {2,3} into {3} — sum over rows\n";
    Backend& be = cpu_backend();
    Tensor a   = Tensor::from_data<double>({1,2,3,4,5,6}, {2,3}, be);
    Tensor dst = Tensor::zeros({3}, be);
    be.kernel_engine()->dispatch_unary(KernelType::ReduceTo, a, dst);
    CHECK_NEAR(dst.at({0}), 5.0, 1e-12);
    CHECK_NEAR(dst.at({1}), 7.0, 1e-12);
    CHECK_NEAR(dst.at({2}), 9.0, 1e-12);
}

void test_reduce_to_stride0_src() {
    std::cout << "[CPUKernel 14] reduce_to: stride-0 broadcast src {2,3} → {1,3}\n";
    Backend& be = cpu_backend();
    Tensor row = Tensor::from_data<double>({10.0, 20.0, 30.0}, {3}, be);
    Tensor bc  = row.view({2, 3}, {0, 1});
    Tensor dst = Tensor::zeros({1, 3}, be);
    be.kernel_engine()->dispatch_unary(KernelType::ReduceTo, bc, dst);
    CHECK_NEAR(dst.at({0, 0}), 20.0, 1e-12);
    CHECK_NEAR(dst.at({0, 1}), 40.0, 1e-12);
    CHECK_NEAR(dst.at({0, 2}), 60.0, 1e-12);
}

void test_matmul_2x3_3x2() {
    std::cout << "[CPUKernel 15] matmul: 2×3 * 3×2\n";
    Backend& be = cpu_backend();
    Tensor a   = Tensor::from_data<double>({1,2,3,4,5,6},    {2,3}, be);
    Tensor b   = Tensor::from_data<double>({7,8,9,10,11,12}, {3,2}, be);
    Tensor out = Tensor::zeros({2, 2}, be);
    be.kernel_engine()->dispatch_matmul(a, b, out);
    CHECK_NEAR(out.at({0,0}),  58.0, 1e-12);
    CHECK_NEAR(out.at({0,1}),  64.0, 1e-12);
    CHECK_NEAR(out.at({1,0}), 139.0, 1e-12);
    CHECK_NEAR(out.at({1,1}), 154.0, 1e-12);
}

void test_contiguous_materialises_view() {
    std::cout << "[CPUKernel 16] contiguous(): non-contiguous view materialised\n";
    Backend& be = cpu_backend();
    Tensor base = Tensor::from_data<double>({1,2,3,4,5,6}, {2,3}, be);
    Tensor view = base.view({3, 2}, {1, 3});
    Tensor cont = view.contiguous();
    CHECK(cont.is_contiguous());
    CHECK_NEAR(cont.at({0,0}), 1.0, 1e-12);
    CHECK_NEAR(cont.at({0,1}), 4.0, 1e-12);
}

void test_neg_contiguous() {
    std::cout << "[CPUKernel 17] neg: {2,3} contiguous\n";
    Backend& be = cpu_backend();
    Tensor a   = Tensor::from_data<double>({1, 2, 3, -4, 0, 6}, {2, 3}, be);
    Tensor out = Tensor::zeros({2, 3}, be);
    be.kernel_engine()->dispatch_unary(KernelType::Neg, a, out);
    CHECK_NEAR(out.at({0,0}), -1.0, 1e-12);
    CHECK_NEAR(out.at({1,0}),  4.0, 1e-12);
}

void test_neg_strided_view() {
    std::cout << "[CPUKernel 18] neg: strided (transposed) view\n";
    Backend& be = cpu_backend();
    Tensor base = Tensor::from_data<double>({1, 2, 3, 4, 5, 6}, {2, 3}, be);
    Tensor view = base.view({3, 2}, {1, 3});
    Tensor out  = Tensor::zeros({3, 2}, be);
    be.kernel_engine()->dispatch_unary(KernelType::Neg, view, out);
    CHECK_NEAR(out.at({0,0}), -1.0, 1e-12);
    CHECK_NEAR(out.at({0,1}), -4.0, 1e-12);
}

// ── Regression tests for fixed bugs ──────────────────────────────────────────

void test_from_data_throws_on_size_mismatch() {
    std::cout << "[CPUKernel 19] from_data throws std::invalid_argument on data size mismatch\n";
    Backend& be = cpu_backend();
    bool threw = false;
    try {
        static_cast<void>(Tensor::from_data<double>({1.0, 2.0, 3.0, 4.0}, {2, 3}, be));
    } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

void test_at_throws_on_undefined_tensor() {
    std::cout << "[CPUKernel 20] at(): undefined tensor throws std::runtime_error\n";
    Tensor t;
    bool threw = false;
    try { static_cast<void>(t.at({0})); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

void test_at_throws_on_oob_index() {
    std::cout << "[CPUKernel 21] at(): out-of-bounds index throws std::out_of_range\n";
    Backend& be = cpu_backend();
    Tensor t = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be);
    bool threw = false;
    try { static_cast<void>(t.at({5})); }
    catch (const std::out_of_range&) { threw = true; }
    CHECK(threw);
}

void test_view_matching_sizes_succeeds() {
    std::cout << "[CPUKernel 22] view(): matching shape/stride sizes — succeeds\n";
    Backend& be = cpu_backend();
    Tensor base = Tensor::from_data<double>({1,2,3,4,5,6}, {2,3}, be);
    Tensor v    = base.view({3, 2}, {1, 3});
    CHECK(v.shape()[0] == 3);
    CHECK(v.shape()[1] == 2);
    CHECK(!v.is_contiguous());
}

void test_cpu_backend_pointer_stability() {
    std::cout << "[CPUKernel 23] cpu_backend() pointer stability\n";
    Backend& a = cpu_backend();
    Backend& b = cpu_backend();
    CHECK(&a == &b);
}

void test_bytes_allocated_zero_after_scope() {
    std::cout << "[CPUKernel 24] bytes_allocated == 0 after all tensors go out of scope\n";
    Backend& be = cpu_backend();
    const std::size_t before = be.memory_manager()->bytes_allocated();
    {
        Tensor a = Tensor::from_data<double>({1,2,3,4}, {2,2}, be);
        Tensor b = Tensor::from_data<double>({5,6,7,8}, {2,2}, be);
        CHECK(be.memory_manager()->bytes_allocated() > before);
    }
    CHECK(be.memory_manager()->bytes_allocated() == before);
}

// ── dispatch_scale / dispatch_axpy / SGD ─────────────────────────────────────

void test_scale_contiguous() {
    std::cout << "[Optim 1] dispatch_scale: contiguous tensor scaled correctly\n";
    Backend& be = cpu_backend();
    Tensor x = Tensor::from_data<double>({1.0, 2.0, 3.0, 4.0}, {4}, be);
    be.kernel_engine()->dispatch_scale(x, 3.0);
    CHECK_NEAR(x.at({0}),  3.0, 1e-12);
    CHECK_NEAR(x.at({3}), 12.0, 1e-12);
}

void test_scale_zero() {
    std::cout << "[Optim 2] dispatch_scale: scale by 0.0 zeros the tensor\n";
    Backend& be = cpu_backend();
    Tensor x = Tensor::from_data<double>({5.0, -3.0, 2.0}, {3}, be);
    be.kernel_engine()->dispatch_scale(x, 0.0);
    for (std::size_t i = 0; i < 3; ++i) CHECK_NEAR(x.at({i}), 0.0, 1e-12);
}

void test_axpy_contiguous() {
    std::cout << "[Optim 3] dispatch_axpy: dst += alpha*src on contiguous tensors\n";
    Backend& be = cpu_backend();
    // dst=[1,2,3], src=[4,5,6], alpha=2.0 → [9, 12, 15]
    Tensor dst = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be);
    Tensor src = Tensor::from_data<double>({4.0, 5.0, 6.0}, {3}, be);
    be.kernel_engine()->dispatch_axpy(dst, 2.0, src);
    CHECK_NEAR(dst.at({0}),  9.0, 1e-12);
    CHECK_NEAR(dst.at({1}), 12.0, 1e-12);
    CHECK_NEAR(dst.at({2}), 15.0, 1e-12);
}

void test_axpy_negative_alpha() {
    std::cout << "[Optim 4] dispatch_axpy: alpha=-1.0 subtracts src from dst\n";
    Backend& be = cpu_backend();
    Tensor dst = Tensor::from_data<double>({10.0, 10.0, 10.0}, {3}, be);
    Tensor src = Tensor::from_data<double>({1.0,  2.0,  3.0},  {3}, be);
    be.kernel_engine()->dispatch_axpy(dst, -1.0, src);
    CHECK_NEAR(dst.at({0}), 9.0, 1e-12);
    CHECK_NEAR(dst.at({2}), 7.0, 1e-12);
}

void test_sgd_basic_step() {
    std::cout << "[Optim 5] SGD: basic step without momentum — param -= lr*grad\n";
    Backend& be = cpu_backend();
    Tensor w = Tensor::from_data<double>({2.0, 4.0, 6.0}, {3}, be, /*requires_grad=*/true);
    Tensor g = Tensor::from_data<double>({1.0, 1.0, 1.0}, {3}, be);
    w.accumulate_grad(g);
    optim::SGD sgd({w}, /*lr=*/0.1);
    sgd.step();
    CHECK_NEAR(w.at({0}), 1.9, 1e-12);
    CHECK_NEAR(w.at({2}), 5.9, 1e-12);
}

void test_sgd_zero_grad() {
    std::cout << "[Optim 6] SGD: zero_grad clears accumulated gradients\n";
    Backend& be = cpu_backend();
    Tensor w = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be, /*requires_grad=*/true);
    Tensor g = Tensor::from_data<double>({5.0, 5.0, 5.0}, {3}, be);
    w.accumulate_grad(g);
    CHECK(w.grad().defined());
    optim::SGD sgd({w}, /*lr=*/0.1);
    sgd.zero_grad();
    CHECK(!w.grad().defined());
}

void test_sgd_momentum() {
    std::cout << "[Optim 7] SGD: momentum — velocity accumulates across steps\n";
    Backend& be = cpu_backend();
    // w=[0], grad=[1] every step, lr=1.0, momentum=0.9
    // Step 1: v=1, w=-1; Step 2: v=1.9, w=-2.9; Step 3: v=2.71, w=-5.61
    Tensor w = Tensor::from_data<double>({0.0}, {1}, be, /*requires_grad=*/true);
    optim::SGD sgd({w}, /*lr=*/1.0, /*momentum=*/0.9);
    auto step_with_grad = [&](double gv) {
        w.zero_grad();
        Tensor gv_t = Tensor::from_data<double>({gv}, {1}, be);
        w.accumulate_grad(gv_t);
        sgd.step();
    };
    step_with_grad(1.0); CHECK_NEAR(w.at({0}), -1.0,  1e-10);
    step_with_grad(1.0); CHECK_NEAR(w.at({0}), -2.9,  1e-10);
    step_with_grad(1.0); CHECK_NEAR(w.at({0}), -5.61, 1e-10);
}

void test_mlp_training_loop() {
    std::cout << "[Optim 8] MLP training loop: loss decreases for 30 SGD steps\n";
    Backend& be = cpu_backend();
    Tensor x = Tensor::from_data<double>(
        {0.1,0.2, 0.3,0.4, 0.5,0.6, 0.7,0.8}, {4, 2}, be);
    Tensor y = Tensor::from_data<double>({0.3, 0.7, 1.1, 1.5}, {4, 1}, be);
    Tensor W1 = Tensor::full({2, 4}, 0.1, be, DType::Float64, /*requires_grad=*/true);
    Tensor b1 = Tensor::full({4},    0.1, be, DType::Float64, /*requires_grad=*/true);
    Tensor W2 = Tensor::full({4, 1}, 0.1, be, DType::Float64, /*requires_grad=*/true);
    Tensor b2 = Tensor::full({1},    0.0, be, DType::Float64, /*requires_grad=*/true);
    optim::SGD sgd({W1, b1, W2, b2}, /*lr=*/0.01);
    double prev_loss = 1e18;
    for (int step = 0; step < 30; ++step) {
        sgd.zero_grad();
        Tensor pre1 = x.matmul(W1);
        Tensor h    = pre1.add(b1.broadcast_to({4, 4})).relu();
        Tensor pre2 = h.matmul(W2);
        Tensor out  = pre2.add(b2.broadcast_to({4, 1}));
        Tensor diff = out.sub(y);
        Tensor loss = diff.mul(diff).mean();
        loss.backward();
        sgd.step();
        double cur_loss = loss.at({0});
        CHECK(cur_loss < prev_loss);
        prev_loss = cur_loss;
    }
}

// ── Runner ────────────────────────────────────────────────────────────────────

void run_cpu_kernel_impl_tests() {
    // Kernel registry tests
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

    // CPUKernelEngine direct dispatch
    test_fill_2d();
    test_fill_scalar();
    test_add_contiguous();
    test_add_broadcast_row();
    test_mul_contiguous();
    test_element_read_scalar();
    test_element_read_2d();
    test_copy_contiguous();
    test_copy_strided_view();
    test_copy_stride0_broadcast();
    test_sum_2d();
    test_sum_noncontiguous();
    test_reduce_to_rows();
    test_reduce_to_stride0_src();
    test_matmul_2x3_3x2();
    test_contiguous_materialises_view();
    test_neg_contiguous();
    test_neg_strided_view();
    test_from_data_throws_on_size_mismatch();
    test_at_throws_on_undefined_tensor();
    test_at_throws_on_oob_index();
    test_view_matching_sizes_succeeds();
    test_cpu_backend_pointer_stability();
    test_bytes_allocated_zero_after_scope();

    // SGD / scale / axpy
    test_scale_contiguous();
    test_scale_zero();
    test_axpy_contiguous();
    test_axpy_negative_alpha();
    test_sgd_basic_step();
    test_sgd_zero_grad();
    test_sgd_momentum();
    test_mlp_training_loop();
}

} // namespace otter::test
