#pragma once

// Debug logging — active only in debug builds (NDEBUG not defined).
//
// OTTER_DBG(fmt, ...) writes a thread-tagged line to stderr.
// stderr is unbuffered on Linux/macOS: output appears immediately even if the
// process crashes on the next line.  This makes it safe to use around suspected
// crash sites without worrying about unflushed buffers.
//
// Usage:
//   OTTER_DBG("accumulate_grad: before lock, tid=%zu", tid);
//   OTTER_DBG("backward: phase %d done", n);

#ifndef NDEBUG
#include <cstdio>
#include <functional>   // std::hash
#include <thread>

namespace otter::detail {
inline std::size_t this_tid() noexcept {
    return std::hash<std::thread::id>{}(std::this_thread::get_id());
}
} // namespace otter::detail

// Writes "OTTER [<tid>] <message>\n" to stderr atomically (one fprintf call).
// The format string and variadic args follow the same rules as printf.
#define OTTER_DBG(fmt, ...)                                                    \
    do {                                                                       \
        std::fprintf(stderr, "OTTER [%016zx] " fmt "\n",                      \
                     ::otter::detail::this_tid(), ##__VA_ARGS__);              \
    } while (0)

#else
// Release build: completely eliminated by the compiler.
#define OTTER_DBG(fmt, ...) ((void)0)
#endif // NDEBUG
