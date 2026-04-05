#include "otter/memory/buffer.h"

#include <cstring>

namespace otter {

Buffer::Buffer(std::size_t bytes, MemoryManager* mm, const void* init_data)
    : size_(bytes), device_(mm->device()), allocator_(mm)
{
    data_ = mm->allocate(bytes); // allocate() returns std::byte* — no cast needed
    if (init_data)
        std::memcpy(data_, init_data, bytes);
    else
        std::memset(data_, 0, bytes);
}

Buffer::~Buffer() noexcept {
    allocator_->free(data_);
}

} // namespace otter
