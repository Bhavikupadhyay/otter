#include "otter/kernel/backend.h"

namespace otter {

Backend::Backend(std::unique_ptr<MemoryManager> mm,
                 std::unique_ptr<KernelEngine>  ke)
    : mm_(std::move(mm)), ke_(std::move(ke))
{}

} // namespace otter
