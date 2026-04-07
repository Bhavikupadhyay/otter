#pragma once

// Private header — included only from src/kernels/ translation units.
// Defines the raw_const<T> / raw_mutable<T> template bodies for KernelEngine.
//
// This header is the only place that simultaneously includes both
// kernel_engine.h and buffer.h with their full definitions. Breaking
// the circular forward-declaration chain:
//   buffer.h         → forward-declares class KernelEngine
//   kernel_engine.h  → forward-declares class Buffer
// Neither includes the other. This file is the bridge.
//
// Include order is intentional: KernelEngine must be fully defined before
// its out-of-class member template bodies are written.

#include "otter/kernel/kernel_engine.h"
#include "otter/memory/buffer.h"
#include "otter/core/passkey.h"  // directly used: Passkey<KernelEngine>{}

namespace otter {

template<typename T>
const T* KernelEngine::raw_const(const Buffer& buf) const noexcept {
    return buf.data<T>(Passkey<KernelEngine>{});
}

template<typename T>
T* KernelEngine::raw_mutable(Buffer& buf) const noexcept {
    return buf.mutable_data<T>(Passkey<KernelEngine>{});
}

} // namespace otter
