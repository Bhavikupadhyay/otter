#pragma once

namespace otter {

// Passkey<T> — compile-time access control idiom.
//
// A method gated on Passkey<T> can only be called by code inside class T,
// because only T can construct Passkey<T> (T is a friend of Passkey<T>).
//
// Zero runtime cost: the constructor is trivial and the object is zero bytes.
// Under -O1 or higher the compiler eliminates it entirely.
//
// Usage:
//   // In buffer.h:
//   const double* data(Passkey<KernelEngine>) const noexcept;
//
//   // In cpu_kernel_engine.cpp (inside KernelEngine subclass):
//   buf->data(Passkey<KernelEngine>{});  // compiles
//
//   // Anywhere else:
//   buf->data(Passkey<KernelEngine>{});  // compile error: Passkey ctor is private

template<typename T>
class Passkey {
    friend T;
    Passkey() = default;
    Passkey(const Passkey&) = default;
};

} // namespace otter
