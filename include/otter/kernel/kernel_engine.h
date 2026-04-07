#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <stdexcept>

namespace otter {

class Buffer;   // forward — full def in src/kernels/dispatcher.h only
class Tensor;   // forward — dispatch signatures only; full def not needed here

// Identifies a kernel op. Both binary and unary ops share this enum.
// Extended as new ops are implemented — zero changes to KernelEngine.
enum class KernelType {
    Add,
    Mul,
    Neg,
    Sum,
    // Sub, Div, Relu, Exp, Log — added as operations are implemented
};

// KernelEngine — concrete kernel dispatch registry.
//
// Holds one Dispatcher per op (or per call-signature family). All polymorphism
// is in the Dispatcher inner classes, not in KernelEngine itself. KernelEngine
// has no pure virtuals — it is not abstract.
//
// Backend constructors (CPUKernelEngine, CUDAKernelEngine) call register_*
// from their constructors to populate the registry. The registry is fixed
// after construction — no modification at runtime.
//
// Op-level registrations (KernelType::Add, KernelType::Mul, ...) must live
// in a separate catalogue .cpp file, not in the constructor. Concretely:
// the constructor calls populate_binary_ops() / populate_unary_ops() which
// are private methods declared in the subclass header but defined in their
// own catalogue file. This keeps the constructor short regardless of op count.
//
// Raw buffer access is gated by Passkey<KernelEngine>. Helpers raw_const<T>
// and raw_mutable<T> are declared protected here; their bodies live in
// src/kernels/dispatcher.h to avoid the circular include:
//   buffer.h forward-declares KernelEngine
//   kernel_engine.h forward-declares Buffer
// Kernel .cpp files must include dispatcher.h, not kernel_engine.h directly.
class KernelEngine {
public:
    // ── Dispatcher bases — one per call-signature family ────────────────────

    // Rule of Five: copy/move deleted on all Dispatcher bases — these are
    // always held via unique_ptr and must never be copied or sliced.
    struct BinaryDispatcher {
        BinaryDispatcher()                                    = default;
        virtual ~BinaryDispatcher()                           = default;
        BinaryDispatcher(const BinaryDispatcher&)             = delete;
        BinaryDispatcher& operator=(const BinaryDispatcher&)  = delete;
        BinaryDispatcher(BinaryDispatcher&&)                  = delete;
        BinaryDispatcher& operator=(BinaryDispatcher&&)       = delete;
        virtual void call(const Tensor& a, const Tensor& b,
                          Tensor& out) const = 0;
    };

    struct UnaryDispatcher {
        UnaryDispatcher()                                     = default;
        virtual ~UnaryDispatcher()                            = default;
        UnaryDispatcher(const UnaryDispatcher&)               = delete;
        UnaryDispatcher& operator=(const UnaryDispatcher&)    = delete;
        UnaryDispatcher(UnaryDispatcher&&)                    = delete;
        UnaryDispatcher& operator=(UnaryDispatcher&&)         = delete;
        virtual void call(const Tensor& a, Tensor& out) const = 0;
    };

    struct FillDispatcher {
        FillDispatcher()                                      = default;
        virtual ~FillDispatcher()                             = default;
        FillDispatcher(const FillDispatcher&)                 = delete;
        FillDispatcher& operator=(const FillDispatcher&)      = delete;
        FillDispatcher(FillDispatcher&&)                      = delete;
        FillDispatcher& operator=(FillDispatcher&&)           = delete;
        virtual void call(Tensor& t, double value) const = 0;
    };

    struct ReduceDispatcher {
        ReduceDispatcher()                                      = default;
        virtual ~ReduceDispatcher()                             = default;
        ReduceDispatcher(const ReduceDispatcher&)               = delete;
        ReduceDispatcher& operator=(const ReduceDispatcher&)    = delete;
        ReduceDispatcher(ReduceDispatcher&&)                    = delete;
        ReduceDispatcher& operator=(ReduceDispatcher&&)         = delete;
        virtual void call(const Tensor& src, Tensor& dst) const = 0;
    };

    struct MatMulDispatcher {
        MatMulDispatcher()                                      = default;
        virtual ~MatMulDispatcher()                             = default;
        MatMulDispatcher(const MatMulDispatcher&)               = delete;
        MatMulDispatcher& operator=(const MatMulDispatcher&)    = delete;
        MatMulDispatcher(MatMulDispatcher&&)                    = delete;
        MatMulDispatcher& operator=(MatMulDispatcher&&)         = delete;
        virtual void call(const Tensor& a, const Tensor& b,
                          Tensor& out) const = 0;
    };

    struct CopyDispatcher {
        CopyDispatcher()                                        = default;
        virtual ~CopyDispatcher()                               = default;
        CopyDispatcher(const CopyDispatcher&)                   = delete;
        CopyDispatcher& operator=(const CopyDispatcher&)        = delete;
        CopyDispatcher(CopyDispatcher&&)                        = delete;
        CopyDispatcher& operator=(CopyDispatcher&&)             = delete;
        virtual void call(const Tensor& src, Tensor& dst) const = 0;
    };

    struct ElementReadDispatcher {
        ElementReadDispatcher()                                          = default;
        virtual ~ElementReadDispatcher()                                 = default;
        ElementReadDispatcher(const ElementReadDispatcher&)              = delete;
        ElementReadDispatcher& operator=(const ElementReadDispatcher&)   = delete;
        ElementReadDispatcher(ElementReadDispatcher&&)                   = delete;
        ElementReadDispatcher& operator=(ElementReadDispatcher&&)        = delete;
        // Returns double for any on-device dtype. On CUDA this is an explicit
        // device→host transfer — never a hidden host dereference.
        [[nodiscard]] virtual double call(const Tensor& t,
                                          std::size_t flat_idx) const = 0;
    };

    // ── Rule of Five ─────────────────────────────────────────────────────────
    KernelEngine()                               = default;
    virtual ~KernelEngine()                      = default;  // virtual: subclasses exist
    KernelEngine(const KernelEngine&)            = delete;
    KernelEngine& operator=(const KernelEngine&) = delete;
    KernelEngine(KernelEngine&&)                 = delete;
    KernelEngine& operator=(KernelEngine&&)      = delete;

    // ── Dispatch entry points — non-virtual; look up registered dispatcher ──

    void dispatch_binary(KernelType op,
                         const Tensor& a, const Tensor& b,
                         Tensor& out) const {
        auto it = binary_.find(op);
        if (it == binary_.end())
            throw std::runtime_error("KernelEngine: no binary dispatcher for op");
        it->second->call(a, b, out);
    }

    void dispatch_unary(KernelType op, const Tensor& a, Tensor& out) const {
        auto it = unary_.find(op);
        if (it == unary_.end())
            throw std::runtime_error("KernelEngine: no unary dispatcher for op");
        it->second->call(a, out);
    }

    void dispatch_fill(Tensor& t, double value) const {
        if (!fill_)
            throw std::runtime_error("KernelEngine: fill dispatcher not registered");
        fill_->call(t, value);
    }

    void dispatch_reduce_to(const Tensor& src, Tensor& dst) const {
        if (!reduce_)
            throw std::runtime_error("KernelEngine: reduce_to dispatcher not registered");
        reduce_->call(src, dst);
    }

    void dispatch_matmul(const Tensor& a, const Tensor& b, Tensor& out) const {
        if (!matmul_)
            throw std::runtime_error("KernelEngine: matmul dispatcher not registered");
        matmul_->call(a, b, out);
    }

    void dispatch_copy(const Tensor& src, Tensor& dst) const {
        if (!copy_)
            throw std::runtime_error("KernelEngine: copy dispatcher not registered");
        copy_->call(src, dst);
    }

    [[nodiscard]] double dispatch_element_read(const Tensor& t,
                                               std::size_t flat_idx) const {
        if (!element_read_)
            throw std::runtime_error(
                "KernelEngine: element_read dispatcher not registered");
        return element_read_->call(t, flat_idx);
    }

protected:
    // ── Registration — called by subclass constructors only ─────────────────

    void register_binary(KernelType op, std::unique_ptr<BinaryDispatcher> d) {
        binary_[op] = std::move(d);
    }
    void register_unary(KernelType op, std::unique_ptr<UnaryDispatcher> d) {
        unary_[op] = std::move(d);
    }
    void register_fill(std::unique_ptr<FillDispatcher> d) {
        fill_ = std::move(d);
    }
    void register_reduce_to(std::unique_ptr<ReduceDispatcher> d) {
        reduce_ = std::move(d);
    }
    void register_matmul(std::unique_ptr<MatMulDispatcher> d) {
        matmul_ = std::move(d);
    }
    void register_copy(std::unique_ptr<CopyDispatcher> d) {
        copy_ = std::move(d);
    }
    void register_element_read(std::unique_ptr<ElementReadDispatcher> d) {
        element_read_ = std::move(d);
    }

    // ── Raw Buffer access — bodies in src/kernels/dispatcher.h ──────────────
    // Kernel .cpp files must include dispatcher.h to get these definitions.
    template<typename T>
    [[nodiscard]] const T* raw_const(const Buffer& buf) const noexcept;

    template<typename T>
    [[nodiscard]] T* raw_mutable(Buffer& buf) const noexcept;

private:
    std::map<KernelType, std::unique_ptr<BinaryDispatcher>> binary_;
    std::map<KernelType, std::unique_ptr<UnaryDispatcher>>  unary_;
    std::unique_ptr<FillDispatcher>        fill_;
    std::unique_ptr<ReduceDispatcher>      reduce_;
    std::unique_ptr<MatMulDispatcher>      matmul_;
    std::unique_ptr<CopyDispatcher>        copy_;
    std::unique_ptr<ElementReadDispatcher> element_read_;
};

} // namespace otter
