#include "otter/tensor.h"

#include <cassert>
#include <stdexcept>

namespace otter {

// ── Private constructor ───────────────────────────────────────────────────────

Tensor::Tensor(std::shared_ptr<Buffer>  buf,
               std::vector<std::size_t> shape,
               std::vector<std::size_t> stride,
               std::size_t              offset,
               DType                    dtype,
               Backend*                 backend)
    : buffer_(std::move(buf))
    , shape_(std::move(shape))
    , stride_(std::move(stride))
    , offset_(offset)
    , dtype_(dtype)
    , backend_(backend)
    , is_contiguous_(stride_ == detail::contiguous_strides(shape_))
{}

// ── Factory ───────────────────────────────────────────────────────────────────

Tensor Tensor::zeros(const std::vector<std::size_t>& shape,
                     Backend& backend,
                     DType dtype)
{
    std::size_t nbytes = dtype_utils::size_of(dtype);
    for (auto d : shape) nbytes *= d;

    auto strides = detail::contiguous_strides(shape);
    auto buf     = std::make_shared<Buffer>(nbytes, backend);  // zero-initialised by Buffer ctor
    return Tensor(std::move(buf), shape, std::move(strides), 0, dtype, &backend);
}

// ── Accessors ─────────────────────────────────────────────────────────────────

std::size_t Tensor::numel() const noexcept {
    std::size_t n = 1;
    for (auto d : shape_) n *= d;
    return n;
}

Backend& Tensor::backend() const {
    assert(defined() && "Tensor::backend() called on undefined tensor");
    return *backend_;
}

Buffer& Tensor::mutable_buffer() {
    assert(defined() && "Tensor::mutable_buffer() called on undefined tensor");
    return *buffer_;
}

const Buffer& Tensor::buffer() const {
    assert(defined() && "Tensor::buffer() called on undefined tensor");
    return *buffer_;
}

// ── Layout ────────────────────────────────────────────────────────────────────

Tensor Tensor::view(std::vector<std::size_t> new_shape,
                    std::vector<std::size_t> new_stride) const
{
    assert(defined() && "Tensor::view() called on undefined tensor");
    return Tensor(buffer_, std::move(new_shape), std::move(new_stride),
                  offset_, dtype_, backend_);
}

Tensor Tensor::contiguous() const {
    assert(defined() && "Tensor::contiguous() called on undefined tensor");
    if (is_contiguous_) return *this;
    // Allocate fresh contiguous buffer and copy elements via the strided copy kernel.
    // The returned tensor is a plain value — no grad history.
    Tensor out = zeros(shape_, *backend_, dtype_);
    backend_->kernel_engine()->dispatch_copy(*this, out);
    return out;
}

// ── Element access ────────────────────────────────────────────────────────────

double Tensor::at(std::initializer_list<std::size_t> indices) const {
    assert(defined() && "Tensor::at() called on undefined tensor");
    assert(indices.size() == shape_.size() && "at(): wrong number of indices");

    std::size_t flat = offset_;
    std::size_t dim  = 0;
    for (std::size_t idx : indices) {
        assert(idx < shape_[dim] && "at(): index out of bounds");
        flat += idx * stride_[dim++];
    }
    // Device-explicit element read — on CUDA this is an explicit device→host transfer.
    return backend_->kernel_engine()->dispatch_element_read(*this, flat);
}

// ── In-place fill ─────────────────────────────────────────────────────────────

void Tensor::fill_(double value) {
    assert(defined()         && "Tensor::fill_() called on undefined tensor");
    assert(is_contiguous_    && "Tensor::fill_() requires a contiguous tensor");
    backend_->kernel_engine()->dispatch_fill(*this, value);
}

} // namespace otter
