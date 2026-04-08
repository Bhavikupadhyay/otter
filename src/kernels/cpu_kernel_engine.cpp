// Include dispatcher.h first — it provides the raw_const / raw_mutable template
// bodies that CPUKernelEngine methods call. Must be before kernel_engine.h is
// pulled in by any subsequent include (dispatcher.h includes it internally).
#include "dispatcher.h"
#include "cpu_kernel_engine.h"

#include "otter/tensor.h"
#include "otter/detail/stride_utils.h"
#include "otter/detail/broadcast.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

namespace otter {

// ─────────────────────────────────────────────────────────────────────────────
// Internal kernel helpers (anonymous namespace — not visible outside .cpp)
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// Returns true if all tensors are contiguous — enables the fast path.
bool all_contiguous(std::initializer_list<const Tensor*> ts) noexcept {
    for (const auto* t : ts)
        if (!t->is_contiguous()) return false;
    return true;
}

// Fast elementwise binary path: all tensors contiguous, pointers pre-offset.
template<typename Op>
void elementwise_contiguous(double* __restrict__ out,
                             const double* __restrict__ a,
                             const double* __restrict__ b,
                             std::size_t n, Op op) noexcept
{
    for (std::size_t i = 0; i < n; ++i) out[i] = op(a[i], b[i]);
}

// General elementwise binary: handles arbitrary strides (including stride-0
// broadcast views). Pointers are offset-adjusted by the caller. The stride
// decomposition iterates logical coordinates starting from 0 relative to each
// tensor's offset, so out[oo] / a[oa] / b[ob] map to the correct elements.
template<typename Op>
void elementwise_strided(double* out, const double* a, const double* b,
                          const std::vector<std::size_t>& shape,
                          const std::vector<std::size_t>& st_out,
                          const std::vector<std::size_t>& st_a,
                          const std::vector<std::size_t>& st_b,
                          Op op) noexcept
{
    const std::size_t ndim = shape.size();
    std::size_t numel = 1;
    for (auto d : shape) numel *= d;

    for (std::size_t flat = 0; flat < numel; ++flat) {
        std::size_t rem = flat, oo = 0, oa = 0, ob = 0;
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            const std::size_t coord = rem % shape[static_cast<std::size_t>(d)];
            rem /= shape[static_cast<std::size_t>(d)];
            oo += coord * st_out[static_cast<std::size_t>(d)];
            oa += coord * st_a  [static_cast<std::size_t>(d)];
            ob += coord * st_b  [static_cast<std::size_t>(d)];
        }
        out[oo] = op(a[oa], b[ob]);
    }
}

// Fast unary elementwise path: contiguous tensors only.
template<typename Op>
void unary_contiguous(double* __restrict__ out,
                      const double* __restrict__ a,
                      std::size_t n, Op op) noexcept
{
    for (std::size_t i = 0; i < n; ++i) out[i] = op(a[i]);
}

// General unary elementwise: handles arbitrary strides.
template<typename Op>
void unary_strided(double* out, const double* a,
                   const std::vector<std::size_t>& shape,
                   const std::vector<std::size_t>& st_out,
                   const std::vector<std::size_t>& st_a,
                   Op op) noexcept
{
    const std::size_t ndim = shape.size();
    std::size_t numel = 1;
    for (auto d : shape) numel *= d;

    for (std::size_t flat = 0; flat < numel; ++flat) {
        std::size_t rem = flat, oo = 0, oa = 0;
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            const std::size_t coord = rem % shape[static_cast<std::size_t>(d)];
            rem /= shape[static_cast<std::size_t>(d)];
            oo += coord * st_out[static_cast<std::size_t>(d)];
            oa += coord * st_a  [static_cast<std::size_t>(d)];
        }
        out[oo] = op(a[oa]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU dispatcher structs — anonymous, only live in this translation unit.
// Each stores a CPUKernelEngine* and forwards call() to the engine's methods.
// ─────────────────────────────────────────────────────────────────────────────

struct CPUFillDispatcher final : KernelEngine::FillDispatcher {
    CPUKernelEngine* engine_;
    explicit CPUFillDispatcher(CPUKernelEngine* e) : engine_(e) {}
    void call(Tensor& t, double v) const override { engine_->cpu_fill(t, v); }
};

struct CPUAddDispatcher final : KernelEngine::BinaryDispatcher {
    CPUKernelEngine* engine_;
    explicit CPUAddDispatcher(CPUKernelEngine* e) : engine_(e) {}
    void call(const Tensor& a, const Tensor& b, Tensor& out) const override {
        engine_->cpu_add(a, b, out);
    }
};

struct CPUMulDispatcher final : KernelEngine::BinaryDispatcher {
    CPUKernelEngine* engine_;
    explicit CPUMulDispatcher(CPUKernelEngine* e) : engine_(e) {}
    void call(const Tensor& a, const Tensor& b, Tensor& out) const override {
        engine_->cpu_mul(a, b, out);
    }
};

struct CPUNegDispatcher final : KernelEngine::UnaryDispatcher {
    CPUKernelEngine* engine_;
    explicit CPUNegDispatcher(CPUKernelEngine* e) : engine_(e) {}
    void call(const Tensor& a, Tensor& out) const override {
        engine_->cpu_neg(a, out);
    }
};

struct CPUSumDispatcher final : KernelEngine::UnaryDispatcher {
    CPUKernelEngine* engine_;
    explicit CPUSumDispatcher(CPUKernelEngine* e) : engine_(e) {}
    void call(const Tensor& a, Tensor& out) const override {
        engine_->cpu_sum(a, out);
    }
};

struct CPUCopyDispatcher final : KernelEngine::CopyDispatcher {
    CPUKernelEngine* engine_;
    explicit CPUCopyDispatcher(CPUKernelEngine* e) : engine_(e) {}
    void call(const Tensor& src, Tensor& dst) const override {
        engine_->cpu_copy(src, dst);
    }
};

struct CPUReduceToDispatcher final : KernelEngine::UnaryDispatcher {
    CPUKernelEngine* engine_;
    explicit CPUReduceToDispatcher(CPUKernelEngine* e) : engine_(e) {}
    void call(const Tensor& src, Tensor& dst) const override {
        engine_->cpu_reduce_to(src, dst);
    }
};

struct CPUMatMulDispatcher final : KernelEngine::MatMulDispatcher {
    CPUKernelEngine* engine_;
    explicit CPUMatMulDispatcher(CPUKernelEngine* e) : engine_(e) {}
    void call(const Tensor& a, const Tensor& b, Tensor& out) const override {
        engine_->cpu_matmul(a, b, out);
    }
};

struct CPUElementReadDispatcher final : KernelEngine::ElementReadDispatcher {
    CPUKernelEngine* engine_;
    explicit CPUElementReadDispatcher(CPUKernelEngine* e) : engine_(e) {}
    [[nodiscard]] double call(const Tensor& t, std::size_t flat_idx) const override {
        return engine_->cpu_element_read(t, flat_idx);
    }
};

struct CPUSubDispatcher final : KernelEngine::BinaryDispatcher {
    CPUKernelEngine* engine_;
    explicit CPUSubDispatcher(CPUKernelEngine* e) : engine_(e) {}
    void call(const Tensor& a, const Tensor& b, Tensor& out) const override {
        engine_->cpu_sub(a, b, out);
    }
};

struct CPUDivDispatcher final : KernelEngine::BinaryDispatcher {
    CPUKernelEngine* engine_;
    explicit CPUDivDispatcher(CPUKernelEngine* e) : engine_(e) {}
    void call(const Tensor& a, const Tensor& b, Tensor& out) const override {
        engine_->cpu_div(a, b, out);
    }
};

struct CPUExpDispatcher final : KernelEngine::UnaryDispatcher {
    CPUKernelEngine* engine_;
    explicit CPUExpDispatcher(CPUKernelEngine* e) : engine_(e) {}
    void call(const Tensor& a, Tensor& out) const override { engine_->cpu_exp(a, out); }
};

struct CPULogDispatcher final : KernelEngine::UnaryDispatcher {
    CPUKernelEngine* engine_;
    explicit CPULogDispatcher(CPUKernelEngine* e) : engine_(e) {}
    void call(const Tensor& a, Tensor& out) const override { engine_->cpu_log(a, out); }
};

struct CPUSqrtDispatcher final : KernelEngine::UnaryDispatcher {
    CPUKernelEngine* engine_;
    explicit CPUSqrtDispatcher(CPUKernelEngine* e) : engine_(e) {}
    void call(const Tensor& a, Tensor& out) const override { engine_->cpu_sqrt(a, out); }
};

struct CPUReluDispatcher final : KernelEngine::UnaryDispatcher {
    CPUKernelEngine* engine_;
    explicit CPUReluDispatcher(CPUKernelEngine* e) : engine_(e) {}
    void call(const Tensor& a, Tensor& out) const override { engine_->cpu_relu(a, out); }
};

struct CPUReluMaskDispatcher final : KernelEngine::UnaryDispatcher {
    CPUKernelEngine* engine_;
    explicit CPUReluMaskDispatcher(CPUKernelEngine* e) : engine_(e) {}
    void call(const Tensor& a, Tensor& out) const override { engine_->cpu_relu_mask(a, out); }
};

} // namespace (anonymous)

// ─────────────────────────────────────────────────────────────────────────────
// CPUKernelEngine — constructor registers all dispatchers
// ─────────────────────────────────────────────────────────────────────────────

CPUKernelEngine::CPUKernelEngine() {
    register_fill        (std::make_unique<CPUFillDispatcher>        (this));
    register_binary      (KernelType::Add,       std::make_unique<CPUAddDispatcher>      (this));
    register_binary      (KernelType::Sub,       std::make_unique<CPUSubDispatcher>      (this));
    register_binary      (KernelType::Mul,       std::make_unique<CPUMulDispatcher>      (this));
    register_binary      (KernelType::Div,       std::make_unique<CPUDivDispatcher>      (this));
    register_unary       (KernelType::Neg,       std::make_unique<CPUNegDispatcher>      (this));
    register_unary       (KernelType::Exp,       std::make_unique<CPUExpDispatcher>      (this));
    register_unary       (KernelType::Log,       std::make_unique<CPULogDispatcher>      (this));
    register_unary       (KernelType::Sqrt,      std::make_unique<CPUSqrtDispatcher>     (this));
    register_unary       (KernelType::Relu,      std::make_unique<CPUReluDispatcher>     (this));
    register_unary       (KernelType::ReluMask,  std::make_unique<CPUReluMaskDispatcher> (this));
    register_unary       (KernelType::ReduceSum, std::make_unique<CPUSumDispatcher>      (this));
    register_unary       (KernelType::ReduceTo,  std::make_unique<CPUReduceToDispatcher> (this));
    register_copy        (std::make_unique<CPUCopyDispatcher>        (this));
    register_matmul      (std::make_unique<CPUMatMulDispatcher>      (this));
    register_element_read(std::make_unique<CPUElementReadDispatcher> (this));
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel implementations
// ─────────────────────────────────────────────────────────────────────────────

void CPUKernelEngine::cpu_fill(Tensor& t, double value) const {
    // fill_ is always called on freshly allocated contiguous tensors.
    // We fill starting at the tensor's offset (0 for all factory outputs).
    double* ptr = raw_mutable<double>(t.mutable_buffer());
    std::fill(ptr + t.offset(), ptr + t.offset() + t.numel(), value);
}

// ── Elementwise binary (add / mul) ────────────────────────────────────────────

void CPUKernelEngine::cpu_add(const Tensor& a, const Tensor& b, Tensor& out) const {
    const double* pa = raw_const<double>(a.buffer())        + a.offset();
    const double* pb = raw_const<double>(b.buffer())        + b.offset();
    double*       po = raw_mutable<double>(out.mutable_buffer()) + out.offset();

    auto add_op = [](double x, double y) noexcept { return x + y; };

    if (all_contiguous({&a, &b, &out})) {
        elementwise_contiguous(po, pa, pb, out.numel(), add_op);
    } else {
        elementwise_strided(po, pa, pb,
                            out.shape(), out.stride(), a.stride(), b.stride(),
                            add_op);
    }
}

void CPUKernelEngine::cpu_mul(const Tensor& a, const Tensor& b, Tensor& out) const {
    const double* pa = raw_const<double>(a.buffer())        + a.offset();
    const double* pb = raw_const<double>(b.buffer())        + b.offset();
    double*       po = raw_mutable<double>(out.mutable_buffer()) + out.offset();

    auto mul_op = [](double x, double y) noexcept { return x * y; };

    if (all_contiguous({&a, &b, &out})) {
        elementwise_contiguous(po, pa, pb, out.numel(), mul_op);
    } else {
        elementwise_strided(po, pa, pb,
                            out.shape(), out.stride(), a.stride(), b.stride(),
                            mul_op);
    }
}

// ── Elementwise unary (neg) ───────────────────────────────────────────────────

void CPUKernelEngine::cpu_neg(const Tensor& a, Tensor& out) const {
    // Both a and out must be contiguous and same shape — caller ensures this.
    assert(a.shape() == out.shape());
    const double* in = raw_const<double>(a.buffer())           + a.offset();
    double*       po = raw_mutable<double>(out.mutable_buffer()) + out.offset();

    if (all_contiguous({&a, &out})) {
        const std::size_t n = a.numel();
        for (std::size_t i = 0; i < n; ++i) po[i] = -in[i];
    } else {
        elementwise_strided(po, in, in,
                            out.shape(), out.stride(), a.stride(), a.stride(),
                            [](double x, double /*unused*/) noexcept { return -x; });
    }
}

// ── Elementwise binary (sub / div) ───────────────────────────────────────────

void CPUKernelEngine::cpu_sub(const Tensor& a, const Tensor& b, Tensor& out) const {
    const double* pa = raw_const<double>(a.buffer())             + a.offset();
    const double* pb = raw_const<double>(b.buffer())             + b.offset();
    double*       po = raw_mutable<double>(out.mutable_buffer()) + out.offset();

    auto sub_op = [](double x, double y) noexcept { return x - y; };

    if (all_contiguous({&a, &b, &out})) {
        elementwise_contiguous(po, pa, pb, out.numel(), sub_op);
    } else {
        elementwise_strided(po, pa, pb,
                            out.shape(), out.stride(), a.stride(), b.stride(),
                            sub_op);
    }
}

void CPUKernelEngine::cpu_div(const Tensor& a, const Tensor& b, Tensor& out) const {
    // IEEE 754: b=0 produces ±inf or nan for Float64 — no check needed.
    const double* pa = raw_const<double>(a.buffer())             + a.offset();
    const double* pb = raw_const<double>(b.buffer())             + b.offset();
    double*       po = raw_mutable<double>(out.mutable_buffer()) + out.offset();

    auto div_op = [](double x, double y) noexcept { return x / y; };

    if (all_contiguous({&a, &b, &out})) {
        elementwise_contiguous(po, pa, pb, out.numel(), div_op);
    } else {
        elementwise_strided(po, pa, pb,
                            out.shape(), out.stride(), a.stride(), b.stride(),
                            div_op);
    }
}

// ── Elementwise unary (exp / log / sqrt / relu / relu_mask) ─────────────────

void CPUKernelEngine::cpu_exp(const Tensor& a, Tensor& out) const {
    assert(a.shape() == out.shape());
    const double* in = raw_const<double>(a.buffer())             + a.offset();
    double*       po = raw_mutable<double>(out.mutable_buffer()) + out.offset();

    auto op = [](double x) noexcept { return std::exp(x); };

    if (all_contiguous({&a, &out})) {
        unary_contiguous(po, in, a.numel(), op);
    } else {
        unary_strided(po, in, out.shape(), out.stride(), a.stride(), op);
    }
}

void CPUKernelEngine::cpu_log(const Tensor& a, Tensor& out) const {
    // IEEE 754: x=0 → -inf, x<0 → nan — no check needed.
    assert(a.shape() == out.shape());
    const double* in = raw_const<double>(a.buffer())             + a.offset();
    double*       po = raw_mutable<double>(out.mutable_buffer()) + out.offset();

    auto op = [](double x) noexcept { return std::log(x); };

    if (all_contiguous({&a, &out})) {
        unary_contiguous(po, in, a.numel(), op);
    } else {
        unary_strided(po, in, out.shape(), out.stride(), a.stride(), op);
    }
}

void CPUKernelEngine::cpu_sqrt(const Tensor& a, Tensor& out) const {
    // IEEE 754: x<0 → nan, x=0 → 0.0 — no check needed.
    assert(a.shape() == out.shape());
    const double* in = raw_const<double>(a.buffer())             + a.offset();
    double*       po = raw_mutable<double>(out.mutable_buffer()) + out.offset();

    auto op = [](double x) noexcept { return std::sqrt(x); };

    if (all_contiguous({&a, &out})) {
        unary_contiguous(po, in, a.numel(), op);
    } else {
        unary_strided(po, in, out.shape(), out.stride(), a.stride(), op);
    }
}

void CPUKernelEngine::cpu_relu(const Tensor& a, Tensor& out) const {
    assert(a.shape() == out.shape());
    const double* in = raw_const<double>(a.buffer())             + a.offset();
    double*       po = raw_mutable<double>(out.mutable_buffer()) + out.offset();

    // At x=0: output is 0.0 (right-hand derivative convention, matches PyTorch).
    auto op = [](double x) noexcept { return x > 0.0 ? x : 0.0; };

    if (all_contiguous({&a, &out})) {
        unary_contiguous(po, in, a.numel(), op);
    } else {
        unary_strided(po, in, out.shape(), out.stride(), a.stride(), op);
    }
}

void CPUKernelEngine::cpu_relu_mask(const Tensor& a, Tensor& out) const {
    // Produces the relu backward mask: 1.0 where input > 0, 0.0 otherwise.
    // At x=0: mask = 0.0 (subgradient 0, matches PyTorch convention).
    assert(a.shape() == out.shape());
    const double* in = raw_const<double>(a.buffer())             + a.offset();
    double*       po = raw_mutable<double>(out.mutable_buffer()) + out.offset();

    auto op = [](double x) noexcept { return x > 0.0 ? 1.0 : 0.0; };

    if (all_contiguous({&a, &out})) {
        unary_contiguous(po, in, a.numel(), op);
    } else {
        unary_strided(po, in, out.shape(), out.stride(), a.stride(), op);
    }
}

// ── Reduce-all (sum) ──────────────────────────────────────────────────────────

void CPUKernelEngine::cpu_sum(const Tensor& a, Tensor& out) const {
    // Caller (SumOperation::forward) is responsible for passing a contiguous
    // input — the kernel must not allocate. Assert enforces the contract.
    assert(a.is_contiguous() && "cpu_sum: input must be contiguous; call contiguous() before dispatch");
    const double* in = raw_const<double>(a.buffer()) + a.offset();

    double s = 0.0;
    const std::size_t n = a.numel();
    for (std::size_t i = 0; i < n; ++i) s += in[i];

    double* dst = raw_mutable<double>(out.mutable_buffer()) + out.offset();
    dst[0] = s;
}

// ── Element read ──────────────────────────────────────────────────────────────

double CPUKernelEngine::cpu_element_read(const Tensor& t, std::size_t flat_idx) const {
    // flat_idx was computed by Tensor::at() including offset + stride indexing.
    return raw_const<double>(t.buffer())[flat_idx];
}

// ── Copy (arbitrary strides → arbitrary strides) ─────────────────────────────
//
// Both src and dst may have any strides. src and dst must have the same shape.
// Used by Tensor::contiguous() (contiguous dst) and SliceOperation backward
// (non-contiguous dst — a view into a zeros tensor at the slice position).

void CPUKernelEngine::cpu_copy(const Tensor& src, Tensor& dst) const {
    assert(src.shape() == dst.shape());

    const double* in  = raw_const<double>(src.buffer())          + src.offset();
    double*       out = raw_mutable<double>(dst.mutable_buffer()) + dst.offset();

    const std::size_t ndim  = src.shape().size();
    const std::size_t numel = src.numel();

    // Odometer-style coordinate iteration: last dimension increments fastest.
    std::vector<std::size_t> coords(ndim, 0);
    for (std::size_t flat = 0; flat < numel; ++flat) {
        std::size_t in_off = 0, out_off = 0;
        for (std::size_t d = 0; d < ndim; ++d) {
            in_off  += coords[d] * src.stride()[d];
            out_off += coords[d] * dst.stride()[d];
        }
        out[out_off] = in[in_off];

        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            if (++coords[static_cast<std::size_t>(d)] < src.shape()[static_cast<std::size_t>(d)])
                break;
            coords[static_cast<std::size_t>(d)] = 0;
        }
    }
}

// ── Reduce-to (scatter-accumulate for broadcast backward) ────────────────────
//
// dst.shape() = original pre-broadcast shape.
// src.shape() = broadcast target shape (gradient coming in).
// dst must be pre-zeroed (use Tensor::zeros before calling).
//
// For each element of src:
//   - decompose flat index to multi-dim coords in src space
//   - drop any prepended dims (in_ndim - out_ndim)
//   - clamp size-1 output dims to coord 0
//   - accumulate src[element] into dst[mapped element]

void CPUKernelEngine::cpu_reduce_to(const Tensor& src, Tensor& dst) const {
    assert(dst.is_contiguous() && "cpu_reduce_to: dst must be contiguous");

    const auto& out_shape = dst.shape();
    const auto& in_shape  = src.shape();
    const std::size_t out_ndim  = out_shape.size();
    const std::size_t in_ndim   = in_shape.size();
    assert(in_ndim >= out_ndim &&
           "cpu_reduce_to: src must have >= dims than dst (src is the broadcast gradient)");

    const double* in  = raw_const<double>(src.buffer())        + src.offset();
    double*       out = raw_mutable<double>(dst.mutable_buffer()) + dst.offset();

    const auto  out_stride = detail::contiguous_strides(out_shape);
    const std::size_t prepended = in_ndim - out_ndim;

    const std::size_t numel_in = src.numel();
    std::vector<std::size_t> coords(in_ndim);

    for (std::size_t flat = 0; flat < numel_in; ++flat) {
        // Decompose flat index into coords of the in tensor.
        std::size_t tmp = flat;
        for (int d = static_cast<int>(in_ndim) - 1; d >= 0; --d) {
            coords[static_cast<std::size_t>(d)] = tmp % in_shape[static_cast<std::size_t>(d)];
            tmp /= in_shape[static_cast<std::size_t>(d)];
        }

        // Compute input element offset using src's actual strides (handles stride-0).
        std::size_t in_off = 0;
        for (std::size_t d = 0; d < in_ndim; ++d) in_off += coords[d] * src.stride()[d];

        // Map to output offset: drop prepended dims, clamp size-1 dims to 0.
        std::size_t out_off = 0;
        for (std::size_t d = prepended; d < in_ndim; ++d) {
            const std::size_t od        = d - prepended;
            const std::size_t out_coord = (out_shape[od] == 1) ? 0 : coords[d];
            out_off += out_coord * out_stride[od];
        }

        out[out_off] += in[in_off];
    }
}

// ── Matrix multiply (basic, no transpose) ────────────────────────────────────
//
// Computes out[...,i,j] = sum_k a[...,i,k] * b[...,k,j]
// a:   [..., M, K] — any strides (batch dims may be stride-0 broadcast views)
// b:   [..., K, N]
// out: [..., M, N] — pre-zeroed, always contiguous
//
// Naive triple loop. Correctness over performance; BLAS can replace this later.

void CPUKernelEngine::cpu_matmul(const Tensor& a, const Tensor& b, Tensor& out) const {
    assert(out.is_contiguous() && "cpu_matmul: out must be contiguous");
    assert(out.shape().size() >= 2 &&
           "cpu_matmul: output tensor must be at least 2-dimensional");
    const double* lhs = raw_const<double>(a.buffer())        + a.offset();
    const double* rhs = raw_const<double>(b.buffer())        + b.offset();
    double*       res = raw_mutable<double>(out.mutable_buffer()) + out.offset();

    const auto& out_shape = out.shape();
    const std::size_t ndim = out_shape.size();
    const std::size_t M    = out_shape[ndim - 2];
    const std::size_t N    = out_shape[ndim - 1];
    const std::size_t K    = a.shape()[ndim - 1];  // no transpose

    const std::size_t lhs_rs = a.stride()[ndim - 2];
    const std::size_t lhs_cs = a.stride()[ndim - 1];
    const std::size_t rhs_rs = b.stride()[ndim - 2];
    const std::size_t rhs_cs = b.stride()[ndim - 1];

    const auto out_stride = detail::contiguous_strides(out_shape);
    const std::size_t out_rs = out_stride[ndim - 2];
    const std::size_t out_cs = out_stride[ndim - 1];

    std::size_t batch = 1;
    for (std::size_t d = 0; d + 2 < ndim; ++d) batch *= out_shape[d];

    std::vector<std::size_t> batch_coords(ndim >= 3 ? ndim - 2 : 0);

    for (std::size_t b_idx = 0; b_idx < batch; ++b_idx) {
        std::size_t tmp = b_idx;
        std::size_t out_base = 0, lhs_base = 0, rhs_base = 0;
        for (int d = static_cast<int>(ndim) - 3; d >= 0; --d) {
            batch_coords[static_cast<std::size_t>(d)] =
                tmp % out_shape[static_cast<std::size_t>(d)];
            tmp /= out_shape[static_cast<std::size_t>(d)];
            out_base  += batch_coords[static_cast<std::size_t>(d)]
                         * out_stride[static_cast<std::size_t>(d)];
            lhs_base  += batch_coords[static_cast<std::size_t>(d)]
                         * a.stride()[static_cast<std::size_t>(d)];
            rhs_base  += batch_coords[static_cast<std::size_t>(d)]
                         * b.stride()[static_cast<std::size_t>(d)];
        }

        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                double acc = 0.0;
                for (std::size_t k = 0; k < K; ++k) {
                    acc += lhs[lhs_base + i * lhs_rs + k * lhs_cs] *
                           rhs[rhs_base + k * rhs_rs + j * rhs_cs];
                }
                res[out_base + i * out_rs + j * out_cs] = acc;
            }
        }
    }
}

} // namespace otter
