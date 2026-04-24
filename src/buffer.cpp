#include "otter/memory/buffer.h"
#include "otter/kernel/backend.h"

#ifdef OTTER_CUDA
#  include "otter/detail/cuda_runtime_mutex.h"
#endif

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
            std::lock_guard<std::mutex> runtime_lock(detail::cuda_runtime_mutex());
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
            std::lock_guard<std::mutex> runtime_lock(detail::cuda_runtime_mutex());
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
