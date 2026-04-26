#include "otter/memory/buffer.h"
#include "otter/kernel/backend.h"

#include <cstring>

#ifdef OTTER_CUDA
#  include "cuda_check.h"
#  include "otter/core/device.h"
#endif

namespace otter {

Buffer::Buffer(std::size_t bytes, Backend& backend, const void* init_data)
    : data_(nullptr), size_(bytes), backend_(&backend)
{
    data_ = backend.memory_manager()->allocate(bytes);
    if (init_data) {
#ifdef OTTER_CUDA
        if (backend.memory_manager()->device() == Device::CUDA) {
            // cudaMemcpy is thread-safe; no external lock needed.
            OTTER_CUDA_CHECK(::cudaMemcpy(data_, init_data, bytes,
                                          cudaMemcpyHostToDevice));
        } else {
            std::memcpy(data_, init_data, bytes);
        }
#else
        std::memcpy(data_, init_data, bytes);
#endif
    } else {
#ifdef OTTER_CUDA
        if (backend.memory_manager()->device() == Device::CUDA) {
            // cudaMemset is thread-safe; no external lock needed.
            OTTER_CUDA_CHECK(::cudaMemset(data_, 0, bytes));
        } else {
            std::memset(data_, 0, bytes);
        }
#else
        std::memset(data_, 0, bytes);
#endif
    }
}

Buffer::~Buffer() noexcept {
    backend_->memory_manager()->free(data_);
}

} // namespace otter
