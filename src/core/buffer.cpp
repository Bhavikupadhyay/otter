#include "otter/memory/buffer.h"
#include "otter/kernel/backend.h"

namespace otter {

Buffer::Buffer(std::size_t bytes, Backend& backend, const void* init_data)
    : data_(nullptr), size_(bytes), backend_(&backend)
{
    data_ = backend.memory_manager()->allocate(bytes);
    try {
        if (init_data)
            backend.memory_manager()->copy_from_host(data_, init_data, bytes);
        else
            backend.memory_manager()->zero_fill(data_, bytes);
    } catch (...) {
        backend_->memory_manager()->free(data_);
        throw;
    }
}

Buffer::~Buffer() noexcept {
    backend_->memory_manager()->free(data_);
}

} // namespace otter
