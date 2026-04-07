#include "test_utils.h"

#include <cstddef>
#include <vector>

#include "otter/tensor.h"
#include "otter/backends/cpu.h"

// All tests use cpu_backend() end-to-end.
// Expected values are computed from the mathematical definition — no reference
// library. CHECK_NEAR tolerance is 1e-12 (double precision; no accumulated error).

namespace otter::test {

// ── Fill ──────────────────────────────────────────────────────────────────────

void test_fill_2d() {
    std::cout << "[CPU 1] fill: {3,4} filled with 7.0 — all 12 elements == 7.0\n";
    Backend& be = cpu_backend();
    Tensor t = Tensor::zeros({3, 4}, be);
    t.fill_(7.0);
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 4; ++c)
            CHECK_NEAR(t.at({r, c}), 7.0, 1e-12);
}

void test_fill_scalar() {
    std::cout << "[CPU 2] fill: {1} with -3.5 — element == -3.5\n";
    Backend& be = cpu_backend();
    Tensor t = Tensor::zeros({1}, be);
    t.fill_(-3.5);
    CHECK_NEAR(t.at({0}), -3.5, 1e-12);
}

// ── Add ───────────────────────────────────────────────────────────────────────

void test_add_contiguous() {
    std::cout << "[CPU 3] add: {2,3} + {2,3} contiguous — element-wise sums correct\n";
    Backend& be = cpu_backend();
    // a = [[1,2,3],[4,5,6]], b = [[10,20,30],[40,50,60]]
    Tensor a = Tensor::from_data<double>({1,2,3,4,5,6},         {2,3}, be);
    Tensor b = Tensor::from_data<double>({10,20,30,40,50,60},   {2,3}, be);
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
    std::cout << "[CPU 4] add: broadcast {3} + {2,3} — stride-0 view + add\n";
    Backend& be = cpu_backend();
    // row = [10, 20, 30]; mat = [[1,2,3],[4,5,6]]
    // broadcast row to {2,3} via stride-0 on dim 0
    Tensor row = Tensor::from_data<double>({10, 20, 30}, {3}, be);
    // view row as {2,3} with strides {0, 1} — stride-0 on first dim
    Tensor row_bc = row.view({2, 3}, {0, 1});
    Tensor mat   = Tensor::from_data<double>({1, 2, 3, 4, 5, 6}, {2, 3}, be);
    Tensor out   = Tensor::zeros({2, 3}, be);
    be.kernel_engine()->dispatch_binary(KernelType::Add, row_bc, mat, out);

    // expected: [[11,22,33],[14,25,36]]
    CHECK_NEAR(out.at({0,0}), 11.0, 1e-12);
    CHECK_NEAR(out.at({0,1}), 22.0, 1e-12);
    CHECK_NEAR(out.at({0,2}), 33.0, 1e-12);
    CHECK_NEAR(out.at({1,0}), 14.0, 1e-12);
    CHECK_NEAR(out.at({1,1}), 25.0, 1e-12);
    CHECK_NEAR(out.at({1,2}), 36.0, 1e-12);
}

// ── Mul ───────────────────────────────────────────────────────────────────────

void test_mul_contiguous() {
    std::cout << "[CPU 5] mul: {2,3} * {2,3} — element-wise products correct\n";
    Backend& be = cpu_backend();
    // a = [[1,2,3],[4,5,6]], b = [[2,3,4],[5,6,7]]
    Tensor a   = Tensor::from_data<double>({1,2,3,4,5,6}, {2,3}, be);
    Tensor b   = Tensor::from_data<double>({2,3,4,5,6,7}, {2,3}, be);
    Tensor out = Tensor::zeros({2, 3}, be);
    be.kernel_engine()->dispatch_binary(KernelType::Mul, a, b, out);

    CHECK_NEAR(out.at({0,0}),  2.0, 1e-12);
    CHECK_NEAR(out.at({0,1}),  6.0, 1e-12);
    CHECK_NEAR(out.at({0,2}), 12.0, 1e-12);
    CHECK_NEAR(out.at({1,0}), 20.0, 1e-12);
    CHECK_NEAR(out.at({1,1}), 30.0, 1e-12);
    CHECK_NEAR(out.at({1,2}), 42.0, 1e-12);
}

void test_mul_broadcast_col() {
    std::cout << "[CPU 6] mul: broadcast {1,3} * {2,3} — correct\n";
    Backend& be = cpu_backend();
    // scale = [[2,3,4]] (shape {1,3}), mat = [[1,1,1],[2,2,2]]
    Tensor scale = Tensor::from_data<double>({2, 3, 4}, {1, 3}, be);
    // view scale as {2,3} with strides {0, 1}
    Tensor scale_bc = scale.view({2, 3}, {0, 1});
    Tensor mat      = Tensor::from_data<double>({1, 1, 1, 2, 2, 2}, {2, 3}, be);
    Tensor out      = Tensor::zeros({2, 3}, be);
    be.kernel_engine()->dispatch_binary(KernelType::Mul, scale_bc, mat, out);

    // expected: [[2,3,4],[4,6,8]]
    CHECK_NEAR(out.at({0,0}), 2.0, 1e-12);
    CHECK_NEAR(out.at({0,1}), 3.0, 1e-12);
    CHECK_NEAR(out.at({0,2}), 4.0, 1e-12);
    CHECK_NEAR(out.at({1,0}), 4.0, 1e-12);
    CHECK_NEAR(out.at({1,1}), 6.0, 1e-12);
    CHECK_NEAR(out.at({1,2}), 8.0, 1e-12);
}

// ── Element read ──────────────────────────────────────────────────────────────

void test_element_read_scalar() {
    std::cout << "[CPU 7] element_read: {1} scalar — returns correct value\n";
    Backend& be = cpu_backend();
    Tensor t = Tensor::from_data<double>({42.0}, {1}, be);
    CHECK_NEAR(t.at({0}), 42.0, 1e-12);
}

void test_element_read_2d() {
    std::cout << "[CPU 8] element_read: {2,3} at (1,2) — returns correct element\n";
    Backend& be = cpu_backend();
    // data = [[10,20,30],[40,50,60]] — element at (1,2) == 60
    Tensor t = Tensor::from_data<double>({10, 20, 30, 40, 50, 60}, {2, 3}, be);
    CHECK_NEAR(t.at({1, 2}), 60.0, 1e-12);
    CHECK_NEAR(t.at({0, 0}), 10.0, 1e-12);
    CHECK_NEAR(t.at({1, 0}), 40.0, 1e-12);
}

// ── Copy ──────────────────────────────────────────────────────────────────────

void test_copy_contiguous() {
    std::cout << "[CPU 9] copy: contiguous → contiguous — values match\n";
    Backend& be = cpu_backend();
    Tensor src = Tensor::from_data<double>({1, 2, 3, 4, 5, 6}, {2, 3}, be);
    Tensor dst = Tensor::zeros({2, 3}, be);
    be.kernel_engine()->dispatch_copy(src, dst);

    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            CHECK_NEAR(dst.at({r, c}), src.at({r, c}), 1e-12);
}

void test_copy_strided_view() {
    std::cout << "[CPU 10] copy: non-contiguous view — correct\n";
    Backend& be = cpu_backend();
    // Original data = [[1,2,3],[4,5,6]] (shape {2,3}, contiguous strides {3,1})
    // View as {3,2} with strides {1,3} — column-major interpretation (transpose-like)
    // Element (r,c) of view reads data at 1*r + 3*c
    // (0,0)=data[0]=1, (0,1)=data[3]=4
    // (1,0)=data[1]=2, (1,1)=data[4]=5
    // (2,0)=data[2]=3, (2,1)=data[5]=6
    Tensor base = Tensor::from_data<double>({1, 2, 3, 4, 5, 6}, {2, 3}, be);
    Tensor view = base.view({3, 2}, {1, 3});

    Tensor dst = Tensor::zeros({3, 2}, be);
    be.kernel_engine()->dispatch_copy(view, dst);

    CHECK_NEAR(dst.at({0,0}), 1.0, 1e-12);
    CHECK_NEAR(dst.at({0,1}), 4.0, 1e-12);
    CHECK_NEAR(dst.at({1,0}), 2.0, 1e-12);
    CHECK_NEAR(dst.at({1,1}), 5.0, 1e-12);
    CHECK_NEAR(dst.at({2,0}), 3.0, 1e-12);
    CHECK_NEAR(dst.at({2,1}), 6.0, 1e-12);
}

void test_copy_stride0_broadcast() {
    std::cout << "[CPU 11] copy: stride-0 broadcast view — materialised correctly\n";
    Backend& be = cpu_backend();
    // row = [10, 20, 30] broadcast to {4,3} via stride {0, 1}
    Tensor row = Tensor::from_data<double>({10, 20, 30}, {3}, be);
    Tensor bc  = row.view({4, 3}, {0, 1});
    Tensor dst = Tensor::zeros({4, 3}, be);
    be.kernel_engine()->dispatch_copy(bc, dst);

    for (std::size_t r = 0; r < 4; ++r) {
        CHECK_NEAR(dst.at({r, 0}), 10.0, 1e-12);
        CHECK_NEAR(dst.at({r, 1}), 20.0, 1e-12);
        CHECK_NEAR(dst.at({r, 2}), 30.0, 1e-12);
    }
}

// ── Sum ───────────────────────────────────────────────────────────────────────

void test_sum_2d() {
    std::cout << "[CPU 12] sum: {3,4} → scalar == sum of all 12 elements\n";
    Backend& be = cpu_backend();
    // Elements 1..12, sum = 78
    Tensor a = Tensor::from_data<double>(
        {1,2,3,4,5,6,7,8,9,10,11,12}, {3, 4}, be);
    Tensor out = Tensor::zeros({1}, be);
    be.kernel_engine()->dispatch_unary(KernelType::Sum, a, out);
    CHECK_NEAR(out.at({0}), 78.0, 1e-12);
}

void test_sum_scalar() {
    std::cout << "[CPU 13] sum: {1} → same scalar\n";
    Backend& be = cpu_backend();
    Tensor a   = Tensor::from_data<double>({5.5}, {1}, be);
    Tensor out = Tensor::zeros({1}, be);
    be.kernel_engine()->dispatch_unary(KernelType::Sum, a, out);
    CHECK_NEAR(out.at({0}), 5.5, 1e-12);
}

void test_sum_noncontiguous() {
    std::cout << "[CPU 14] sum: non-contiguous view → same result as contiguous() + sum\n";
    Backend& be = cpu_backend();
    // base = [[1,2,3],[4,5,6]], view as {3,2} with strides {1,3}
    // values seen: 1,4, 2,5, 3,6 → sum = 21
    Tensor base = Tensor::from_data<double>({1,2,3,4,5,6}, {2,3}, be);
    Tensor view = base.view({3, 2}, {1, 3});
    Tensor out  = Tensor::zeros({1}, be);
    be.kernel_engine()->dispatch_unary(KernelType::Sum, view, out);
    CHECK_NEAR(out.at({0}), 21.0, 1e-12);
}

// ── Reduce-to ─────────────────────────────────────────────────────────────────

void test_reduce_to_rows() {
    std::cout << "[CPU 15] reduce_to: {2,3} into {3} — sum over rows correct\n";
    Backend& be = cpu_backend();
    // a = [[1,2,3],[4,5,6]] → sum cols: [5,7,9]
    Tensor a   = Tensor::from_data<double>({1,2,3,4,5,6}, {2,3}, be);
    Tensor dst = Tensor::zeros({3}, be);  // pre-zeroed
    be.kernel_engine()->dispatch_reduce_to(a, dst);
    CHECK_NEAR(dst.at({0}), 5.0, 1e-12);
    CHECK_NEAR(dst.at({1}), 7.0, 1e-12);
    CHECK_NEAR(dst.at({2}), 9.0, 1e-12);
}

void test_reduce_to_identity() {
    std::cout << "[CPU 16] reduce_to: {1,3} into {1,3} — identity (no reduction needed)\n";
    Backend& be = cpu_backend();
    Tensor a   = Tensor::from_data<double>({7.0, 8.0, 9.0}, {1, 3}, be);
    Tensor dst = Tensor::zeros({1, 3}, be);
    be.kernel_engine()->dispatch_reduce_to(a, dst);
    CHECK_NEAR(dst.at({0,0}), 7.0, 1e-12);
    CHECK_NEAR(dst.at({0,1}), 8.0, 1e-12);
    CHECK_NEAR(dst.at({0,2}), 9.0, 1e-12);
}

void test_reduce_to_scalar() {
    std::cout << "[CPU 17] reduce_to: {2,3} into {1} — sum all elements\n";
    Backend& be = cpu_backend();
    // sum 1+2+3+4+5+6 = 21
    Tensor a   = Tensor::from_data<double>({1,2,3,4,5,6}, {2,3}, be);
    Tensor dst = Tensor::zeros({1}, be);
    be.kernel_engine()->dispatch_reduce_to(a, dst);
    CHECK_NEAR(dst.at({0}), 21.0, 1e-12);
}

// ── MatMul ────────────────────────────────────────────────────────────────────

void test_matmul_2x3_3x2() {
    std::cout << "[CPU 18] matmul: 2×3 * 3×2 — result matches hand calc\n";
    Backend& be = cpu_backend();
    // a = [[1,2,3],[4,5,6]], b = [[7,8],[9,10],[11,12]]
    // out[0,0] = 1*7+2*9+3*11 = 7+18+33 = 58
    // out[0,1] = 1*8+2*10+3*12 = 8+20+36 = 64
    // out[1,0] = 4*7+5*9+6*11 = 28+45+66 = 139
    // out[1,1] = 4*8+5*10+6*12 = 32+50+72 = 154
    Tensor a   = Tensor::from_data<double>({1,2,3,4,5,6},    {2,3}, be);
    Tensor b   = Tensor::from_data<double>({7,8,9,10,11,12}, {3,2}, be);
    Tensor out = Tensor::zeros({2, 2}, be);
    be.kernel_engine()->dispatch_matmul(a, b, out);
    CHECK_NEAR(out.at({0,0}),  58.0, 1e-12);
    CHECK_NEAR(out.at({0,1}),  64.0, 1e-12);
    CHECK_NEAR(out.at({1,0}), 139.0, 1e-12);
    CHECK_NEAR(out.at({1,1}), 154.0, 1e-12);
}

void test_matmul_identity() {
    std::cout << "[CPU 19] matmul: 3×3 identity * A — result equals A\n";
    Backend& be = cpu_backend();
    // identity 3x3
    Tensor I = Tensor::from_data<double>(
        {1,0,0, 0,1,0, 0,0,1}, {3,3}, be);
    // A = [[2,3],[5,7],[11,13]]
    Tensor A   = Tensor::from_data<double>({2,3,5,7,11,13}, {3,2}, be);
    Tensor out = Tensor::zeros({3, 2}, be);
    be.kernel_engine()->dispatch_matmul(I, A, out);
    CHECK_NEAR(out.at({0,0}),  2.0, 1e-12);
    CHECK_NEAR(out.at({0,1}),  3.0, 1e-12);
    CHECK_NEAR(out.at({1,0}),  5.0, 1e-12);
    CHECK_NEAR(out.at({1,1}),  7.0, 1e-12);
    CHECK_NEAR(out.at({2,0}), 11.0, 1e-12);
    CHECK_NEAR(out.at({2,1}), 13.0, 1e-12);
}

// ── contiguous() ─────────────────────────────────────────────────────────────

void test_contiguous_materialises_view() {
    std::cout << "[CPU 20] contiguous(): non-contiguous view materialised — values preserved\n";
    Backend& be = cpu_backend();
    // base = [[1,2,3],[4,5,6]], transposed view {3,2} strides {1,3}
    // contiguous copy should be [[1,4],[2,5],[3,6]]
    Tensor base  = Tensor::from_data<double>({1,2,3,4,5,6}, {2,3}, be);
    Tensor view  = base.view({3, 2}, {1, 3});
    Tensor cont  = view.contiguous();

    CHECK(cont.is_contiguous());
    CHECK(cont.shape()[0] == 3);
    CHECK(cont.shape()[1] == 2);
    CHECK_NEAR(cont.at({0,0}), 1.0, 1e-12);
    CHECK_NEAR(cont.at({0,1}), 4.0, 1e-12);
    CHECK_NEAR(cont.at({1,0}), 2.0, 1e-12);
    CHECK_NEAR(cont.at({1,1}), 5.0, 1e-12);
    CHECK_NEAR(cont.at({2,0}), 3.0, 1e-12);
    CHECK_NEAR(cont.at({2,1}), 6.0, 1e-12);
}

// ── Backend identity + memory tracking ───────────────────────────────────────

void test_cpu_backend_pointer_stability() {
    std::cout << "[CPU 21] cpu_backend() pointer stability — same Backend& returned every call\n";
    Backend& a = cpu_backend();
    Backend& b = cpu_backend();
    CHECK(&a == &b);
}

void test_bytes_allocated_zero_after_scope() {
    std::cout << "[CPU 22] bytes_allocated == 0 after all tensors go out of scope\n";
    Backend& be = cpu_backend();
    // Force a fresh allocation count baseline by reading current state.
    // (cpu_backend is a singleton — it may have prior allocations from other
    //  tests that are still alive. We check only that allocations made HERE
    //  are fully released when the tensors leave scope.)
    const std::size_t before = be.memory_manager()->bytes_allocated();
    {
        Tensor a = Tensor::from_data<double>({1,2,3,4}, {2,2}, be);
        Tensor b = Tensor::from_data<double>({5,6,7,8}, {2,2}, be);
        // Both allocations are live
        CHECK(be.memory_manager()->bytes_allocated() > before);
    }
    // Both shared_ptrs dropped — buffers freed
    CHECK(be.memory_manager()->bytes_allocated() == before);
}

// ── Runner ────────────────────────────────────────────────────────────────────

void run_cpu_kernel_tests() {
    test_fill_2d();
    test_fill_scalar();
    test_add_contiguous();
    test_add_broadcast_row();
    test_mul_contiguous();
    test_mul_broadcast_col();
    test_element_read_scalar();
    test_element_read_2d();
    test_copy_contiguous();
    test_copy_strided_view();
    test_copy_stride0_broadcast();
    test_sum_2d();
    test_sum_scalar();
    test_sum_noncontiguous();
    test_reduce_to_rows();
    test_reduce_to_identity();
    test_reduce_to_scalar();
    test_matmul_2x3_3x2();
    test_matmul_identity();
    test_contiguous_materialises_view();
    test_cpu_backend_pointer_stability();
    test_bytes_allocated_zero_after_scope();
}

} // namespace otter::test
