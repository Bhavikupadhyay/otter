#include "otter/memory/buffer.h"
#include "otter/kernel/backend.h"

#include <cstring>

namespace otter {

Buffer::Buffer(std::size_t bytes, Backend& backend, const void* init_data)
    : size_(bytes), backend_(&backend)
{
    data_ = backend.memory_manager()->allocate(bytes);
    if (init_data)
        std::memcpy(data_, init_data, bytes);
    else
        std::memset(data_, 0, bytes);
}

Buffer::~Buffer() noexcept {
    backend_->memory_manager()->free(data_);
}

} // namespace otter
