#include "test_utils.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "otter/backends/cpu.h"
#include "otter/backends/cuda.h"
#include "otter/autograd/no_grad_guard.h"
#include "otter/tensor.h"

// Private header — accessible because tests/CMakeLists.txt adds PROJECT_SOURCE_DIR/src.
// Only used for the stream raw-handle test.
#include "backends/cuda_stream.h"

namespace otter::test {

// ─────────────────────────────────────────────────────────────────────────────
// Helper: try a call, return true if it threw std::runtime_error whose
// what() contains `fragment`.
// ─────────────────────────────────────────────────────────────────────────────

namespace {

template<typename F>
bool throws_with(F&& f, const char* fragment, std::string& out_msg) {
    try { f(); return false; }
    catch (const std::runtime_error& e) {
        out_msg = e.what();
        return std::string(e.what()).find(fragment) != std::string::npos;
    }
    catch (...) { return false; }
}

} // namespace


// ═════════════════════════════════════════════════════════════════════════════
// A — Memory / allocator
// ═════════════════════════════════════════════════════════════════════════════

void test_cuda_A1_bytes_allocated_starts_at_zero() {
    std::cout << "[CUDA A1] bytes_allocated is 0 when no tensors are live\n";
    // After all previous (A*) tests have run, all temps are out of scope.
    CHECK(cuda_backend().memory_manager()->bytes_allocated() == 0);
}

void test_cuda_A2_allocate_increments_free_decrements() {
    std::cout << "[CUDA A2] allocate increments bytes_allocated; free returns it to 0\n";
    const std::size_t before = cuda_backend().memory_manager()->bytes_allocated();
    {
        Tensor t = Tensor::zeros({16}, cuda_backend());
        CHECK(cuda_backend().memory_manager()->bytes_allocated() > before);
    }
    CHECK(cuda_backend().memory_manager()->bytes_allocated() == before);
}

void test_cuda_A3_multiple_tensors_accumulate_and_release() {
    std::cout << "[CUDA A3] multiple tensors accumulate bytes; all released on destruct\n";
    const std::size_t start = cuda_backend().memory_manager()->bytes_allocated();
    {
        Tensor a = Tensor::zeros({4},  cuda_backend());
        Tensor b = Tensor::zeros({8},  cuda_backend());
        Tensor c = Tensor::zeros({16}, cuda_backend());
        // At minimum 4+8+16 = 28 doubles = 224 bytes extra
        CHECK(cuda_backend().memory_manager()->bytes_allocated()
              >= start + 28 * sizeof(double));
    }
    CHECK(cuda_backend().memory_manager()->bytes_allocated() == start);
}

void test_cuda_A4_release_cache_does_not_crash() {
    std::cout << "[CUDA A4] release_cache() (== cudaDeviceSynchronize) does not crash\n";
    cuda_backend().memory_manager()->release_cache();
    CHECK(cuda_backend().memory_manager()->bytes_allocated() == 0);
}

void test_cuda_A5_bytes_reserved_equals_allocated() {
    std::cout << "[CUDA A5] bytes_reserved() == bytes_allocated() — no pool overhead\n";
    Tensor t = Tensor::zeros({32}, cuda_backend());
    CHECK(cuda_backend().memory_manager()->bytes_reserved() ==
          cuda_backend().memory_manager()->bytes_allocated());
}


// ═════════════════════════════════════════════════════════════════════════════
// B — Tensor creation
// ═════════════════════════════════════════════════════════════════════════════

void test_cuda_B1_zeros_all_zero() {
    std::cout << "[CUDA B1] zeros({4}) — all elements 0.0\n";
    Tensor t = Tensor::zeros({4}, cuda_backend());
    CHECK(t.defined());
    CHECK(t.numel() == 4);
    for (std::size_t i = 0; i < 4; ++i)
        CHECK_NEAR(t.at({i}), 0.0, 1e-12);
}

void test_cuda_B2_ones_all_one() {
    std::cout << "[CUDA B2] ones({4}) — all elements 1.0\n";
    Tensor t = Tensor::ones({4}, cuda_backend());
    for (std::size_t i = 0; i < 4; ++i)
        CHECK_NEAR(t.at({i}), 1.0, 1e-12);
}

void test_cuda_B3_full_constant_value() {
    std::cout << "[CUDA B3] full({4}, 7.5) — all elements 7.5\n";
    Tensor t = Tensor::full({4}, 7.5, cuda_backend());
    for (std::size_t i = 0; i < 4; ++i)
        CHECK_NEAR(t.at({i}), 7.5, 1e-12);
}

void test_cuda_B4_from_data_1d() {
    std::cout << "[CUDA B4] from_data 1D — values read back correctly\n";
    Tensor t = Tensor::from_data<double>({1.1, 2.2, 3.3, 4.4}, {4}, cuda_backend());
    CHECK_NEAR(t.at({0}), 1.1, 1e-10);
    CHECK_NEAR(t.at({1}), 2.2, 1e-10);
    CHECK_NEAR(t.at({2}), 3.3, 1e-10);
    CHECK_NEAR(t.at({3}), 4.4, 1e-10);
}

void test_cuda_B5_from_data_2d() {
    std::cout << "[CUDA B5] from_data 2D — stride-based at({r,c}) correct\n";
    Tensor t = Tensor::from_data<double>({1,2,3,4,5,6}, {2,3}, cuda_backend());
    CHECK(t.shape() == (std::vector<std::size_t>{2, 3}));
    CHECK_NEAR(t.at({0,0}), 1.0, 1e-12);
    CHECK_NEAR(t.at({0,1}), 2.0, 1e-12);
    CHECK_NEAR(t.at({0,2}), 3.0, 1e-12);
    CHECK_NEAR(t.at({1,0}), 4.0, 1e-12);
    CHECK_NEAR(t.at({1,1}), 5.0, 1e-12);
    CHECK_NEAR(t.at({1,2}), 6.0, 1e-12);
}

void test_cuda_B6_zeros_like() {
    std::cout << "[CUDA B6] zeros_like — same shape, all 0.0, on CUDA\n";
    Tensor t = Tensor::from_data<double>({1,2,3,4}, {2,2}, cuda_backend());
    Tensor z = Tensor::zeros_like(t);
    CHECK(z.backend().device() == Device::CUDA);
    CHECK(z.shape() == t.shape());
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 2; ++c)
            CHECK_NEAR(z.at({r, c}), 0.0, 1e-12);
}

void test_cuda_B7_ones_like() {
    std::cout << "[CUDA B7] ones_like — same shape, all 1.0, on CUDA\n";
    Tensor t = Tensor::zeros({3}, cuda_backend());
    Tensor o = Tensor::ones_like(t);
    CHECK(o.backend().device() == Device::CUDA);
    for (std::size_t i = 0; i < 3; ++i)
        CHECK_NEAR(o.at({i}), 1.0, 1e-12);
}

void test_cuda_B8_requires_grad_on_cuda() {
    std::cout << "[CUDA B8] requires_grad=true on CUDA: flag set, is_leaf, grad undefined\n";
    Tensor t = Tensor::zeros({4}, cuda_backend(), DType::Float64, /*requires_grad=*/true);
    CHECK(t.requires_grad());
    CHECK(t.is_leaf());
    CHECK(!t.grad().defined());
}


// ═════════════════════════════════════════════════════════════════════════════
// C — fill_ and element access
// ═════════════════════════════════════════════════════════════════════════════

void test_cuda_C1_fill_basic() {
    std::cout << "[CUDA C1] fill_(3.14) — kernel writes, at() reads back\n";
    Tensor t = Tensor::zeros({8}, cuda_backend());
    t.fill_(3.14);
    for (std::size_t i = 0; i < 8; ++i)
        CHECK_NEAR(t.at({i}), 3.14, 1e-10);
}

void test_cuda_C2_fill_negative() {
    std::cout << "[CUDA C2] fill_(-2.5) — negative value written correctly\n";
    Tensor t = Tensor::zeros({4}, cuda_backend());
    t.fill_(-2.5);
    for (std::size_t i = 0; i < 4; ++i)
        CHECK_NEAR(t.at({i}), -2.5, 1e-12);
}

void test_cuda_C3_fill_zero_overwrites() {
    std::cout << "[CUDA C3] fill_(0.0) overwrites non-zero content\n";
    Tensor t = Tensor::full({4}, 9.9, cuda_backend());
    t.fill_(0.0);
    for (std::size_t i = 0; i < 4; ++i)
        CHECK_NEAR(t.at({i}), 0.0, 1e-12);
}

void test_cuda_C4_fill_twice_second_wins() {
    std::cout << "[CUDA C4] fill_ twice: second value wins\n";
    Tensor t = Tensor::zeros({4}, cuda_backend());
    t.fill_(1.0);
    t.fill_(2.0);
    for (std::size_t i = 0; i < 4; ++i)
        CHECK_NEAR(t.at({i}), 2.0, 1e-12);
}

void test_cuda_C5_at_2d_stride_correct() {
    std::cout << "[CUDA C5] at({r,c}) on 2D tensor: stride-based index correct\n";
    // 3×4 tensor: value at logical [r][c] = r*10 + c
    std::vector<double> vals;
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 4; ++c)
            vals.push_back(static_cast<double>(r * 10 + c));
    Tensor t = Tensor::from_data<double>(vals, {3,4}, cuda_backend());
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 4; ++c)
            CHECK_NEAR(t.at({r, c}), static_cast<double>(r * 10 + c), 1e-12);
}

void test_cuda_C6_large_tensor_fill_and_spot_check() {
    std::cout << "[CUDA C6] 1024-element fill + first/mid/last check\n";
    constexpr std::size_t N = 1024;
    Tensor t = Tensor::zeros({N}, cuda_backend());
    t.fill_(6.28);
    CHECK_NEAR(t.at({0}),     6.28, 1e-10);
    CHECK_NEAR(t.at({N/2}),   6.28, 1e-10);
    CHECK_NEAR(t.at({N - 1}), 6.28, 1e-10);
}


// ═════════════════════════════════════════════════════════════════════════════
// D — to_vector() and print()
// ═════════════════════════════════════════════════════════════════════════════

void test_cuda_D1_to_vector_1d() {
    std::cout << "[CUDA D1] to_vector<double>() on 1D CUDA tensor\n";
    Tensor t = Tensor::from_data<double>({10.0, 20.0, 30.0}, {3}, cuda_backend());
    auto v = t.to_vector<double>();
    CHECK(v.size() == 3);
    CHECK_NEAR(v[0], 10.0, 1e-12);
    CHECK_NEAR(v[1], 20.0, 1e-12);
    CHECK_NEAR(v[2], 30.0, 1e-12);
}

void test_cuda_D2_to_vector_2d() {
    std::cout << "[CUDA D2] to_vector<double>() on 2D CUDA tensor — row-major order\n";
    Tensor t = Tensor::from_data<double>({1,2,3,4,5,6}, {2,3}, cuda_backend());
    auto v = t.to_vector<double>();
    CHECK(v.size() == 6);
    for (std::size_t i = 0; i < 6; ++i)
        CHECK_NEAR(v[i], static_cast<double>(i + 1), 1e-12);
}

void test_cuda_D3_print_does_not_throw() {
    std::cout << "[CUDA D3] print() on CUDA tensor: no crash\n";
    Tensor t = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, cuda_backend());
    bool threw = false;
    try { t.print("cuda_test"); } catch (...) { threw = true; }
    CHECK(!threw);
}


// ═════════════════════════════════════════════════════════════════════════════
// E — Views on CUDA (pure metadata — no kernel dispatch)
// ═════════════════════════════════════════════════════════════════════════════

void test_cuda_E1_reshape_flatten() {
    std::cout << "[CUDA E1] reshape {2,3} → {6}: values preserved\n";
    Tensor t = Tensor::from_data<double>({1,2,3,4,5,6}, {2,3}, cuda_backend());
    Tensor r = t.reshape({6});
    CHECK(r.shape() == (std::vector<std::size_t>{6}));
    for (std::size_t i = 0; i < 6; ++i)
        CHECK_NEAR(r.at({i}), static_cast<double>(i + 1), 1e-12);
}

void test_cuda_E2_reshape_reinterpret() {
    std::cout << "[CUDA E2] reshape {2,3} → {3,2}: row-major reinterpretation\n";
    Tensor t = Tensor::from_data<double>({1,2,3,4,5,6}, {2,3}, cuda_backend());
    Tensor r = t.reshape({3, 2});
    CHECK(r.shape() == (std::vector<std::size_t>{3, 2}));
    CHECK_NEAR(r.at({0,0}), 1.0, 1e-12);
    CHECK_NEAR(r.at({0,1}), 2.0, 1e-12);
    CHECK_NEAR(r.at({1,0}), 3.0, 1e-12);
    CHECK_NEAR(r.at({1,1}), 4.0, 1e-12);
    CHECK_NEAR(r.at({2,0}), 5.0, 1e-12);
    CHECK_NEAR(r.at({2,1}), 6.0, 1e-12);
}

void test_cuda_E3_reshape_numel_mismatch_throws() {
    std::cout << "[CUDA E3] reshape numel mismatch throws\n";
    Tensor t = Tensor::zeros({2,3}, cuda_backend());
    bool threw = false;
    try { (void)t.reshape({2,4}); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

void test_cuda_E4_transpose_values_correct() {
    std::cout << "[CUDA E4] transpose(0,1) on CUDA tensor: shape {3,2}, values t[j][i]\n";
    Tensor a = Tensor::from_data<double>({1,2,3,4,5,6}, {2,3}, cuda_backend());
    Tensor t = a.transpose(0, 1);
    CHECK(t.shape() == (std::vector<std::size_t>{3, 2}));
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            CHECK_NEAR(t.at({i, j}), a.at({j, i}), 1e-12);
}

void test_cuda_E5_transpose_dim_out_of_range_throws() {
    std::cout << "[CUDA E5] transpose dim out of range throws\n";
    Tensor t = Tensor::zeros({2,3}, cuda_backend());
    bool threw = false;
    try { (void)t.transpose(0, 5); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

void test_cuda_E6_transpose_is_view_shares_buffer() {
    std::cout << "[CUDA E6] transpose shares buffer — is_unique false on both\n";
    Tensor a = Tensor::from_data<double>({1,2,3,4}, {2,2}, cuda_backend());
    Tensor t = a.transpose(0, 1);
    CHECK(!a.is_unique());
    CHECK(!t.is_unique());
}

void test_cuda_E7_transpose_is_not_contiguous() {
    std::cout << "[CUDA E7] transpose result is not contiguous\n";
    Tensor a = Tensor::from_data<double>({1,2,3,4}, {2,2}, cuda_backend());
    CHECK(a.is_contiguous());
    Tensor t = a.transpose(0, 1);
    CHECK(!t.is_contiguous());
}

void test_cuda_E8_contiguous_on_noncontiguous_cuda_materialises() {
    std::cout << "[CUDA E8] contiguous() on non-contiguous CUDA tensor: "
                 "produces correct contiguous copy\n";
    // a = [[1,2,3],[4,5,6]] shape {2,3}
    // nc = a^T = [[1,4],[2,5],[3,6]] shape {3,2}, non-contiguous
    Tensor a  = Tensor::from_data<double>({1,2,3,4,5,6}, {2,3}, cuda_backend());
    Tensor nc = a.transpose(0, 1);
    CHECK(!nc.is_contiguous());
    Tensor c = nc.contiguous();
    CHECK(c.is_contiguous());
    CHECK(c.shape() == (std::vector<std::size_t>{3, 2}));
    // Logical values of nc (= a^T): nc[i][j] = a[j][i]
    CHECK_NEAR(c.at({0,0}), nc.at({0,0}), 1e-12);  // a[0][0] = 1
    CHECK_NEAR(c.at({0,1}), nc.at({0,1}), 1e-12);  // a[1][0] = 4
    CHECK_NEAR(c.at({1,0}), nc.at({1,0}), 1e-12);  // a[0][1] = 2
    CHECK_NEAR(c.at({1,1}), nc.at({1,1}), 1e-12);  // a[1][1] = 5
    CHECK_NEAR(c.at({2,0}), nc.at({2,0}), 1e-12);  // a[0][2] = 3
    CHECK_NEAR(c.at({2,1}), nc.at({2,1}), 1e-12);  // a[1][2] = 6
}


// ═════════════════════════════════════════════════════════════════════════════
// F — Cross-device copy edge cases
// ═════════════════════════════════════════════════════════════════════════════

void test_cuda_F1_cpu_to_cuda_to_cpu_round_trip() {
    std::cout << "[CUDA F1] CPU → cuda() → cpu() round-trip: values preserved\n";
    Tensor h = Tensor::from_data<double>({1.1, 2.2, 3.3, 4.4}, {4}, cpu_backend());
    Tensor d = h.cuda();
    Tensor h2 = d.cpu();
    CHECK(h2.backend().device() == Device::CPU);
    CHECK_NEAR(h2.at({0}), 1.1, 1e-10);
    CHECK_NEAR(h2.at({1}), 2.2, 1e-10);
    CHECK_NEAR(h2.at({2}), 3.3, 1e-10);
    CHECK_NEAR(h2.at({3}), 4.4, 1e-10);
}

void test_cuda_F2_2d_tensor_round_trip() {
    std::cout << "[CUDA F2] 2D {2,3} CPU → cuda → cpu: at({r,c}) preserved\n";
    std::vector<double> vals = {1,2,3,4,5,6};
    Tensor h  = Tensor::from_data<double>(vals, {2,3}, cpu_backend());
    Tensor d  = h.cuda();
    Tensor h2 = d.cpu();
    CHECK(h2.shape() == (std::vector<std::size_t>{2, 3}));
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            CHECK_NEAR(h2.at({r, c}), vals[r * 3 + c], 1e-12);
}

void test_cuda_F3_noncontiguous_cpu_to_cuda() {
    std::cout << "[CUDA F3] non-contiguous CPU tensor → cuda(): CPU contiguous() "
                 "runs first, values correct on CUDA\n";
    // h = [[1,2],[3,4]]; ht = h^T = [[1,3],[2,4]] (non-contiguous on CPU)
    Tensor h  = Tensor::from_data<double>({1,2,3,4}, {2,2}, cpu_backend());
    Tensor ht = h.transpose(0, 1);
    CHECK(!ht.is_contiguous());

    Tensor d = ht.cuda();  // CPU dispatch_copy → contiguous; then cudaMemcpy
    CHECK(d.backend().device() == Device::CUDA);
    CHECK(d.is_contiguous());
    // Logical values of ht: [0][0]=1, [0][1]=3, [1][0]=2, [1][1]=4
    CHECK_NEAR(d.at({0,0}), ht.at({0,0}), 1e-12);
    CHECK_NEAR(d.at({0,1}), ht.at({0,1}), 1e-12);
    CHECK_NEAR(d.at({1,0}), ht.at({1,0}), 1e-12);
    CHECK_NEAR(d.at({1,1}), ht.at({1,1}), 1e-12);
}

void test_cuda_F4_noncontiguous_cuda_to_cpu_works() {
    std::cout << "[CUDA F4] non-contiguous CUDA tensor → cpu(): "
                 "CUDA copy kernel materialises before transfer\n";
    // t = [[1,2,3],[4,5,6]] shape {2,3}
    // nc = t^T shape {3,2}, non-contiguous on CUDA
    Tensor t  = Tensor::from_data<double>({1,2,3,4,5,6}, {2,3}, cuda_backend());
    Tensor nc = t.transpose(0, 1);
    CHECK(!nc.is_contiguous());
    Tensor h = nc.cpu();
    CHECK(h.backend().device() == Device::CPU);
    CHECK(h.is_contiguous());
    CHECK(h.shape() == (std::vector<std::size_t>{3, 2}));
    // Values must match logical transpose: h[i][j] == t[j][i]
    CHECK_NEAR(h.at({0,0}), nc.at({0,0}), 1e-12);  // t[0][0] = 1
    CHECK_NEAR(h.at({0,1}), nc.at({0,1}), 1e-12);  // t[1][0] = 4
    CHECK_NEAR(h.at({1,0}), nc.at({1,0}), 1e-12);  // t[0][1] = 2
    CHECK_NEAR(h.at({1,1}), nc.at({1,1}), 1e-12);  // t[1][1] = 5
    CHECK_NEAR(h.at({2,0}), nc.at({2,0}), 1e-12);  // t[0][2] = 3
    CHECK_NEAR(h.at({2,1}), nc.at({2,1}), 1e-12);  // t[1][2] = 6
}

void test_cuda_F5_to_cuda_on_cuda_is_fast_path() {
    std::cout << "[CUDA F5] to(Device::CUDA) on CUDA tensor is the identity fast path\n";
    const std::size_t before = cuda_backend().memory_manager()->bytes_allocated();
    Tensor t = Tensor::zeros({8}, cuda_backend());
    t.fill_(5.0);
    const std::size_t with_t = cuda_backend().memory_manager()->bytes_allocated();
    Tensor same = t.to(Device::CUDA);
    // Fast path: no new allocation
    CHECK(cuda_backend().memory_manager()->bytes_allocated() == with_t);
    CHECK_NEAR(same.at({0}), 5.0, 1e-12);
    (void)before;
}

void test_cuda_F6_to_cpu_on_cpu_is_fast_path() {
    std::cout << "[CUDA F6] to(Device::CPU) on CPU tensor is the identity fast path\n";
    Tensor t = Tensor::from_data<double>({1,2,3}, {3}, cpu_backend());
    Tensor same = t.to(Device::CPU);
    CHECK(same.backend().device() == Device::CPU);
    CHECK_NEAR(same.at({0}), 1.0, 1e-12);
}

void test_cuda_F7_to_does_not_preserve_requires_grad() {
    std::cout << "[CUDA F7] to() drops autograd history — returned tensor is a plain leaf\n";
    Tensor h = Tensor::from_data<double>({1,2,3}, {3}, cpu_backend(),
                                         /*requires_grad=*/true);
    CHECK(h.requires_grad());
    Tensor d = h.cuda();
    CHECK(!d.requires_grad());  // to() always returns a plain leaf
    Tensor h2 = d.cpu();
    CHECK(!h2.requires_grad());
}

void test_cuda_F8_bytes_allocated_restored_after_transfer() {
    std::cout << "[CUDA F8] CUDA bytes_allocated returns to baseline after transfer tensor "
                 "is destroyed\n";
    const std::size_t base = cuda_backend().memory_manager()->bytes_allocated();
    {
        Tensor h = Tensor::from_data<double>({1,2,3,4}, {4}, cpu_backend());
        Tensor d = h.cuda();
        CHECK(cuda_backend().memory_manager()->bytes_allocated() > base);
    }
    CHECK(cuda_backend().memory_manager()->bytes_allocated() == base);
}

void test_cuda_F9_large_tensor_transfer() {
    std::cout << "[CUDA F9] 512-element: CPU fill → cuda() → cpu() spot-check\n";
    constexpr std::size_t N = 512;
    std::vector<double> vals(N);
    for (std::size_t i = 0; i < N; ++i) vals[i] = static_cast<double>(i) * 0.5;
    Tensor h  = Tensor::from_data<double>(vals, {N}, cpu_backend());
    Tensor d  = h.cuda();
    Tensor h2 = d.cpu();
    CHECK_NEAR(h2.at({0}),     vals[0],     1e-10);
    CHECK_NEAR(h2.at({N/4}),   vals[N/4],   1e-10);
    CHECK_NEAR(h2.at({N/2}),   vals[N/2],   1e-10);
    CHECK_NEAR(h2.at({N - 1}), vals[N - 1], 1e-10);
}


// ═════════════════════════════════════════════════════════════════════════════
// G — Unregistered dispatchers throw with predictable messages
// ═════════════════════════════════════════════════════════════════════════════

void test_cuda_G1_add_computes_correctly() {
    std::cout << "[CUDA G1] add() on CUDA tensors: {1,2,3,4} + {5,6,7,8} = {6,8,10,12}\n";
    Tensor a = Tensor::from_data<double>({1,2,3,4}, {4}, cuda_backend());
    Tensor b = Tensor::from_data<double>({5,6,7,8}, {4}, cuda_backend());
    Tensor c = a.add(b);
    CHECK(c.shape() == (std::vector<std::size_t>{4}));
    CHECK_NEAR(c.at({0}),  6.0, 1e-12);
    CHECK_NEAR(c.at({1}),  8.0, 1e-12);
    CHECK_NEAR(c.at({2}), 10.0, 1e-12);
    CHECK_NEAR(c.at({3}), 12.0, 1e-12);
}

void test_cuda_G2_mul_computes_correctly() {
    std::cout << "[CUDA G2] mul() on CUDA tensors: {2,3,4,5} * {3,4,5,6} = {6,12,20,30}\n";
    Tensor a = Tensor::from_data<double>({2,3,4,5}, {4}, cuda_backend());
    Tensor b = Tensor::from_data<double>({3,4,5,6}, {4}, cuda_backend());
    Tensor c = a.mul(b);
    CHECK(c.shape() == (std::vector<std::size_t>{4}));
    CHECK_NEAR(c.at({0}),  6.0, 1e-12);
    CHECK_NEAR(c.at({1}), 12.0, 1e-12);
    CHECK_NEAR(c.at({2}), 20.0, 1e-12);
    CHECK_NEAR(c.at({3}), 30.0, 1e-12);
}

void test_cuda_G3_neg_computes_correctly() {
    std::cout << "[CUDA G3] neg() on CUDA tensor: {1,-2,3,-4} → {-1,2,-3,4}\n";
    Tensor a = Tensor::from_data<double>({1,-2,3,-4}, {4}, cuda_backend());
    Tensor c = a.neg();
    CHECK_NEAR(c.at({0}), -1.0, 1e-12);
    CHECK_NEAR(c.at({1}),  2.0, 1e-12);
    CHECK_NEAR(c.at({2}), -3.0, 1e-12);
    CHECK_NEAR(c.at({3}),  4.0, 1e-12);
}

void test_cuda_G4_exp_computes_correctly() {
    std::cout << "[CUDA G4] exp() on CUDA tensor: exp({0,1}) = {1.0, e}\n";
    Tensor a = Tensor::from_data<double>({0.0, 1.0}, {2}, cuda_backend());
    Tensor c = a.exp();
    // exp(0) = 1.0, exp(1) = e ≈ 2.71828...
    CHECK_NEAR(c.at({0}), 1.0,             1e-10);
    CHECK_NEAR(c.at({1}), 2.718281828459,  1e-9);
}

void test_cuda_G5_sum_computes_correctly() {
    std::cout << "[CUDA G5] sum() on CUDA tensor: sum({1,2,3,4}) = 10.0\n";
    Tensor a = Tensor::from_data<double>({1,2,3,4}, {4}, cuda_backend());
    Tensor s = a.sum();
    CHECK(s.numel() == 1);
    // 1+2+3+4 = 10
    CHECK_NEAR(s.at({0}), 10.0, 1e-10);
}

void test_cuda_G6_matmul_computes_correctly() {
    std::cout << "[CUDA G6] matmul() on CUDA tensors: [[1,2],[3,4]] @ [[5,6],[7,8]]\n";
    Tensor a = Tensor::from_data<double>({1,2,3,4}, {2,2}, cuda_backend());
    Tensor b = Tensor::from_data<double>({5,6,7,8}, {2,2}, cuda_backend());
    Tensor c = a.matmul(b);
    CHECK(c.shape() == (std::vector<std::size_t>{2, 2}));
    // [0][0] = 1*5 + 2*7 = 19
    // [0][1] = 1*6 + 2*8 = 22
    // [1][0] = 3*5 + 4*7 = 43
    // [1][1] = 3*6 + 4*8 = 50
    CHECK_NEAR(c.at({0,0}), 19.0, 1e-10);
    CHECK_NEAR(c.at({0,1}), 22.0, 1e-10);
    CHECK_NEAR(c.at({1,0}), 43.0, 1e-10);
    CHECK_NEAR(c.at({1,1}), 50.0, 1e-10);
}

void test_cuda_G7_relu_computes_correctly() {
    std::cout << "[CUDA G7] relu() on CUDA tensor: {-1,0,2,-3} → {0,0,2,0}\n";
    Tensor a = Tensor::from_data<double>({-1,0,2,-3}, {4}, cuda_backend());
    Tensor c = a.relu();
    // At x=0: output is 0.0 (right-hand derivative convention, matches PyTorch).
    CHECK_NEAR(c.at({0}), 0.0, 1e-12);
    CHECK_NEAR(c.at({1}), 0.0, 1e-12);
    CHECK_NEAR(c.at({2}), 2.0, 1e-12);
    CHECK_NEAR(c.at({3}), 0.0, 1e-12);
}

void test_cuda_G8_to_on_undefined_throws() {
    std::cout << "[CUDA G8] to(Device::CUDA) on undefined tensor throws\n";
    Tensor undef;
    std::string msg;
    bool ok = throws_with([&]{ (void)undef.to(Device::CUDA); },
                          "Tensor::to() called on undefined tensor", msg);
    CHECK(ok);
}


// ═════════════════════════════════════════════════════════════════════════════
// H — Streams
// ═════════════════════════════════════════════════════════════════════════════

void test_cuda_H1_default_stream_not_null() {
    std::cout << "[CUDA H1] default_stream() returns non-null\n";
    CHECK(cuda_backend().default_stream() != nullptr);
}

void test_cuda_H2_stream_raw_handle_valid() {
    std::cout << "[CUDA H2] CUDAStream::raw() returns a valid (non-null) cudaStream_t\n";
    Stream* s = cuda_backend().default_stream();
    auto*   cs = static_cast<CUDAStream*>(s);
    CHECK(cs != nullptr);
    CHECK(cs->raw() != nullptr);
}


// ═════════════════════════════════════════════════════════════════════════════
// I — Autograd metadata and buffer sharing (mirrors test_tensor.cpp)
// ═════════════════════════════════════════════════════════════════════════════

void test_cuda_I1_zeros_default_no_grad() {
    std::cout << "[CUDA I1] zeros: default — leaf, requires_grad=false, grad undefined\n";
    Tensor t = Tensor::zeros({2,3}, cuda_backend());
    CHECK(!t.requires_grad());
    CHECK(t.is_leaf());
    CHECK(!t.grad().defined());
}

void test_cuda_I2_zeros_with_grad() {
    std::cout << "[CUDA I2] zeros: requires_grad=true — leaf, grad undefined\n";
    Tensor t = Tensor::zeros({3}, cuda_backend(), DType::Float64, true);
    CHECK(t.requires_grad());
    CHECK(t.is_leaf());
    CHECK(!t.grad().defined());
}

void test_cuda_I3_from_data_with_grad_values_correct() {
    std::cout << "[CUDA I3] from_data: requires_grad=true, values correct, grad undefined\n";
    Tensor t = Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, cuda_backend(), true);
    CHECK(t.requires_grad());
    CHECK(t.is_leaf());
    CHECK_NEAR(t.at({0}), 1.0, 1e-12);
    CHECK_NEAR(t.at({1}), 2.0, 1e-12);
    CHECK_NEAR(t.at({2}), 3.0, 1e-12);
    CHECK(!t.grad().defined());
}

void test_cuda_I4_detach_clears_grad_same_shape() {
    std::cout << "[CUDA I4] detach(): clears requires_grad, shape preserved, values readable\n";
    Tensor t = Tensor::zeros({2,3}, cuda_backend(), DType::Float64, true);
    Tensor d = t.detach();
    CHECK(!d.requires_grad());
    CHECK(!d.grad().defined());
    CHECK(d.shape() == t.shape());
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            CHECK_NEAR(d.at({r, c}), 0.0, 1e-12);
}

void test_cuda_I5_is_unique_fresh_tensor() {
    std::cout << "[CUDA I5] is_unique() true for freshly allocated CUDA tensor\n";
    Tensor t = Tensor::zeros({4}, cuda_backend());
    CHECK(t.is_unique());
}

void test_cuda_I6_is_unique_false_after_copy_restored_after_destroy() {
    std::cout << "[CUDA I6] is_unique false after copy; restored when copy destroyed\n";
    Tensor t = Tensor::zeros({4}, cuda_backend());
    {
        Tensor u = t;
        CHECK(!t.is_unique());
        CHECK(!u.is_unique());
    }
    CHECK(t.is_unique());
}

void test_cuda_I7_is_unique_false_for_reshape_view() {
    std::cout << "[CUDA I7] reshape view shares buffer — is_unique false on both\n";
    Tensor t = Tensor::zeros({2,3}, cuda_backend());
    Tensor v = t.reshape({6});
    CHECK(!t.is_unique());
    CHECK(!v.is_unique());
}


// ═════════════════════════════════════════════════════════════════════════════
// Runner
// ═════════════════════════════════════════════════════════════════════════════

void run_cuda_tests() {
    // A — Memory
    test_cuda_A1_bytes_allocated_starts_at_zero();
    test_cuda_A2_allocate_increments_free_decrements();
    test_cuda_A3_multiple_tensors_accumulate_and_release();
    test_cuda_A4_release_cache_does_not_crash();
    test_cuda_A5_bytes_reserved_equals_allocated();

    // B — Tensor creation
    test_cuda_B1_zeros_all_zero();
    test_cuda_B2_ones_all_one();
    test_cuda_B3_full_constant_value();
    test_cuda_B4_from_data_1d();
    test_cuda_B5_from_data_2d();
    test_cuda_B6_zeros_like();
    test_cuda_B7_ones_like();
    test_cuda_B8_requires_grad_on_cuda();

    // C — fill_ and element access
    test_cuda_C1_fill_basic();
    test_cuda_C2_fill_negative();
    test_cuda_C3_fill_zero_overwrites();
    test_cuda_C4_fill_twice_second_wins();
    test_cuda_C5_at_2d_stride_correct();
    test_cuda_C6_large_tensor_fill_and_spot_check();

    // D — to_vector() and print()
    test_cuda_D1_to_vector_1d();
    test_cuda_D2_to_vector_2d();
    test_cuda_D3_print_does_not_throw();

    // E — Views on CUDA
    test_cuda_E1_reshape_flatten();
    test_cuda_E2_reshape_reinterpret();
    test_cuda_E3_reshape_numel_mismatch_throws();
    test_cuda_E4_transpose_values_correct();
    test_cuda_E5_transpose_dim_out_of_range_throws();
    test_cuda_E6_transpose_is_view_shares_buffer();
    test_cuda_E7_transpose_is_not_contiguous();
    test_cuda_E8_contiguous_on_noncontiguous_cuda_materialises();

    // F — Cross-device copy edge cases
    test_cuda_F1_cpu_to_cuda_to_cpu_round_trip();
    test_cuda_F2_2d_tensor_round_trip();
    test_cuda_F3_noncontiguous_cpu_to_cuda();
    test_cuda_F4_noncontiguous_cuda_to_cpu_works();
    test_cuda_F5_to_cuda_on_cuda_is_fast_path();
    test_cuda_F6_to_cpu_on_cpu_is_fast_path();
    test_cuda_F7_to_does_not_preserve_requires_grad();
    test_cuda_F8_bytes_allocated_restored_after_transfer();
    test_cuda_F9_large_tensor_transfer();

    // G — Compute correctness (kernels registered)
    test_cuda_G1_add_computes_correctly();
    test_cuda_G2_mul_computes_correctly();
    test_cuda_G3_neg_computes_correctly();
    test_cuda_G4_exp_computes_correctly();
    test_cuda_G5_sum_computes_correctly();
    test_cuda_G6_matmul_computes_correctly();
    test_cuda_G7_relu_computes_correctly();
    test_cuda_G8_to_on_undefined_throws();

    // H — Streams
    test_cuda_H1_default_stream_not_null();
    test_cuda_H2_stream_raw_handle_valid();

    // I — Autograd metadata (mirrors test_tensor.cpp)
    test_cuda_I1_zeros_default_no_grad();
    test_cuda_I2_zeros_with_grad();
    test_cuda_I3_from_data_with_grad_values_correct();
    test_cuda_I4_detach_clears_grad_same_shape();
    test_cuda_I5_is_unique_fresh_tensor();
    test_cuda_I6_is_unique_false_after_copy_restored_after_destroy();
    test_cuda_I7_is_unique_false_for_reshape_view();
}

} // namespace otter::test
