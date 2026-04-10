#pragma once

namespace otter {

// Stream — abstract execution context for a device.
//
// On synchronous backends (CPU) there is no stream concept; default_stream()
// returns nullptr and all operations execute in the caller's thread.
//
// On asynchronous backends (CUDA), a Stream wraps a device queue
// (e.g. cudaStream_t). Operations submitted to the same Stream execute in
// submission order; operations on different Streams may execute concurrently.
// Synchronisation points between Streams are expressed via events (future work).
//
// Rule of Five: non-copyable, non-movable. Always heap-allocated and returned
// as a raw non-owning pointer from Backend::default_stream(). Lifetime is tied
// to the Backend that created it.
class Stream {
public:
    virtual ~Stream() = default;

    Stream(const Stream&)            = delete;
    Stream& operator=(const Stream&) = delete;
    Stream(Stream&&)                 = delete;
    Stream& operator=(Stream&&)      = delete;

protected:
    Stream() = default;
};

} // namespace otter
