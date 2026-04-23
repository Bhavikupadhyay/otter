#pragma once

// Element-wise functor types for CUDA and CPU kernel templates.
//
// Each functor provides a single static __host__ __device__ apply() method.
// Using static apply() (rather than operator()) allows the kernel to call
// F::apply(...) without passing a functor instance through register args.
//
// Binary functors: apply(double a, double b) -> double
// Unary functors:  apply(double x) -> double

struct AddFunctor {
    static __host__ __device__ double apply(double a, double b) noexcept { return a + b; }
};

struct SubFunctor {
    static __host__ __device__ double apply(double a, double b) noexcept { return a - b; }
};

struct MulFunctor {
    static __host__ __device__ double apply(double a, double b) noexcept { return a * b; }
};

struct DivFunctor {
    // IEEE 754: b=0 produces ±inf or nan — no check needed.
    static __host__ __device__ double apply(double a, double b) noexcept { return a / b; }
};

struct NegFunctor {
    static __host__ __device__ double apply(double x) noexcept { return -x; }
};

struct ExpFunctor {
    static __host__ __device__ double apply(double x) noexcept { return ::exp(x); }
};

struct LogFunctor {
    // IEEE 754: x=0 → -inf, x<0 → nan — no check needed.
    static __host__ __device__ double apply(double x) noexcept { return ::log(x); }
};

struct SqrtFunctor {
    // IEEE 754: x<0 → nan — no check needed.
    static __host__ __device__ double apply(double x) noexcept { return ::sqrt(x); }
};

struct ReluFunctor {
    // At x=0: output is 0.0 (right-hand derivative convention, matches PyTorch).
    static __host__ __device__ double apply(double x) noexcept { return x > 0.0 ? x : 0.0; }
};

struct ReluMaskFunctor {
    // 1.0 where input > 0, 0.0 otherwise. At x=0: mask=0.0 (matches PyTorch).
    static __host__ __device__ double apply(double x) noexcept { return x > 0.0 ? 1.0 : 0.0; }
};
