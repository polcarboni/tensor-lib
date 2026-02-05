#include "core/TensorImpl.hpp"

namespace tensor {

    // ---------------------------------------- HELPER FUNCTIONS ----------------------------------------

    void TensorImpl::refresh_metadata() {
        // TODO: recompute strides, total_size, etc.
    }

    size_t TensorImpl::get_physical_offset(const std::vector<size_t>& indices) const {
        // TODO: compute linear offset from multidimensional indices
        return 0;
    }

    // ---------------------------------------- CONSTRUCTOR ----------------------------------------

    TensorImpl::TensorImpl(const std::vector<size_t>& shape, ScalarType dtype, Device device, const void* src)
        : shape_(shape), dtype_(dtype), device_(device)
    {
        // TODO: initialize storage_ with proper size based on shape and dtype
        // TODO: copy data from src if provided
        refresh_metadata();
    }

    // ---------------------------------------- GEOMETRIC FUNCTIONS ----------------------------------------

    bool TensorImpl::is_contiguous() const {
        // TODO: check if strides_ correspond to contiguous memory layout
        return false;
    }

    std::unique_ptr<TensorImpl> TensorImpl::clone() {
        // TODO: deep copy
        return nullptr;
    }

    std::unique_ptr<TensorImpl> TensorImpl::contiguous() const {
        // TODO: return contiguous copy if not already contiguous
        return nullptr;
    }

    std::unique_ptr<TensorImpl> TensorImpl::view(std::vector<size_t>& new_shape) const {
        // TODO: create a view with new shape, sharing the same storage
        return nullptr;
    }

} // namespace tensor
