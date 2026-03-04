#include "core/Types.hpp"
#include "core/TensorImpl.hpp"
#include "core/Dispatchers.hpp"
#include "ops/OpsRegistry.hpp"
#include <cassert>
#include <cstddef>


/**
 * TODO: check which templates require explicit instantiations
 */

namespace tensor {

    // -------------------------------------------------------------------------------------------------------------  
    //                                                  GETTERS
    // -------------------------------------------------------------------------------------------------------------        

    std::vector<size_t>& TensorImpl::get_shape()   { return shape_; }
    std::vector<size_t>& TensorImpl::get_strides() { return strides_; }
    ScalarType TensorImpl::get_dtype()             { return dtype_; }
    size_t TensorImpl::get_total_size()            { return total_size_; }
    Device TensorImpl::get_device()                { return device_; }
    bool TensorImpl::requires_grad()               { return requires_grad_; }
    bool TensorImpl::get_contiguous()              { return contiguous_; }


    void TensorImpl::set_shape(const std::vector<size_t>& shape)     { shape_ = shape; }
    void TensorImpl::set_strides(const std::vector<size_t>& strides) { strides_ = strides; }
    void TensorImpl::set_dtype(ScalarType dtype)                     { dtype_ = dtype; }
    void TensorImpl::set_total_size(size_t total_size)               { total_size_ = total_size; }
    void TensorImpl::set_device(const Device& device)                { device_ = device; }
    void TensorImpl::set_requires_grad(bool requires_grad)           { requires_grad_ = requires_grad; }
    void TensorImpl::set_contiguous(bool contiguous)                 { contiguous_ = contiguous; }


    // -------------------------------------------------------------------------------------------------------------  
    //                                              HELPER FUNCTIONS
    // -------------------------------------------------------------------------------------------------------------

    void TensorImpl::refresh_metadata()
    {
        // The function computes the values of: total_size_ and strides_ based on shape_ 
        if (shape_.empty()) {
            total_size_ = (storage_ == nullptr) ? 0 : 1;
            strides_.clear();
            contiguous_ = true;
            return;
        }
        
        // Compute total_size_
        total_size_ = 1;
        for (size_t dim : shape_)
            total_size_ *= dim;
        
        // Compute strides_
        if (strides_.empty()) {
            strides_.resize(shape_.size());
            strides_.back() = 1;
            for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
                strides_[i] = strides_[i + 1] * shape_[i + 1];
            }
        } else {
            if (strides_.size() != shape_.size())
                throw std::runtime_error("TensorImpl::refresh_metadata: strides and shape size mismatch");
        }

        contiguous_ = is_contiguous();
    }


    size_t TensorImpl::get_physical_offset(const std::vector<size_t>& indices) const
    {
        assert(indices.size() == shape_.size());

        size_t physical_offset = offset_;
        
        for (size_t i = 0; i < indices.size(); ++i) {
            physical_offset += indices[i] * strides_[i];
        }

        return physical_offset;
    }


    // -------------------------------------------------------------------------------------------------------------  
    //                                                  CONSTRUCTORS
    // -------------------------------------------------------------------------------------------------------------
    
    TensorImpl::TensorImpl() = default;
    TensorImpl::~TensorImpl() = default;

    TensorImpl::TensorImpl(const TensorImpl& other)
        : storage_(other.storage_), dtype_(other.dtype_), device_(other.device_), offset_(other.offset_),
            shape_(other.shape_), strides_(other.strides_), total_size_(other.total_size_),
            contiguous_(other.contiguous_), requires_grad_(other.requires_grad_) 
    {
        if (other.autograd_meta_)
            // Placeholder (empty autograd meta), should call the AutogradMeta copy constructor instead.
            autograd_meta_ = std::make_unique<grad::AutogradMeta>();
    }

    TensorImpl& TensorImpl::operator=(const TensorImpl& other)
    {
        if (this != &other) {
            TensorImpl temp(other);
            std::swap(*this, temp);
        }
        return *this;
    }

    TensorImpl::TensorImpl(TensorImpl&& other) noexcept = default;
    TensorImpl& TensorImpl::operator=(TensorImpl&& other) noexcept = default;


    // ------------------------------------ CONSTRUCTOR OVERLOADS -------------------------------------

    TensorImpl::TensorImpl(const std::vector<size_t>& shape,
                           ScalarType dtype,
                           Device device,
                           bool requires_grad,
                           void* src)
        : shape_(std::move(shape)), dtype_(dtype), device_(std::move(device)), requires_grad_(requires_grad)
    {
        refresh_metadata();

        if(total_size_ > 0) {
            storage_ = std::make_shared<Storage>(total_size_ * element_size(dtype_), device_);
        } else {
            storage_ = nullptr;
        }

        if (requires_grad_) {
            autograd_meta_ = std::make_unique<grad::AutogradMeta>();
        }

        if (src) {
            throw std::runtime_error("REFACTORED: TO BE IMPLEMENTED");
        }
    }
    
    TensorImpl::TensorImpl(const std::vector<size_t>& shape,
                double fill_value,
                ScalarType dtype,
                Device device,
                bool requires_grad)
        : TensorImpl(shape, dtype, device, requires_grad)
    {
        fill_const(fill_value);
    }
    
    
    // -------------------------------------------------------------------------------------------------------------  
    //                                              INDEXERS/ACCESSORS
    // -------------------------------------------------------------------------------------------------------------

    // Data accessor helper
    template <typename T>
    T* TensorImpl::data_ptr()
    {
        if (!storage_) return nullptr;
        return reinterpret_cast<T*>(static_cast<char*>(storage_->data()) + (offset_ * element_size(dtype_)));
    }

    void* TensorImpl::data_ptr() {
        if (!storage_) return nullptr;
        return static_cast<char*>(storage_->data()) + (offset_ * element_size(dtype_));
    }

    const void* TensorImpl::data_ptr() const
    {
        if (!storage_) return nullptr;
        return static_cast<const char*>(storage_->data()) + (offset_ * element_size(dtype_));
    }

    template <typename T>
    const T* TensorImpl::data_ptr() const
    {
        return static_cast<const T*>(data_ptr());
    }

    // TODO: provide const version
    template <typename T>
    T& TensorImpl::operator()(const std::initializer_list<size_t>& indices)
    {
        if (indices.size() != shape_.size())
            throw std::runtime_error("operator(): Index dimension mismatch");

        size_t idx = get_physical_offset(indices);
        return static_cast<T*>(storage_->data())[idx];
    }

    std::shared_ptr<TensorImpl> TensorImpl::operator[](size_t index)
    {
        if (shape_.empty()) {
            throw std::runtime_error("operator[]: cannot index a 0-dim tensor.");
        }

        if (index >= shape_[0]) {
            throw std::out_of_range("Index out of range for dimension 0.");
        }

        // INCOMPLETE
        return std::make_shared<TensorImpl>();  // placeholder return
    }


    // -------------------------------------------------------------------------------------------------------------  
    //                                          FILLING OPERATIONS
    // -------------------------------------------------------------------------------------------------------------

    void TensorImpl::fill_const(double value)
    {
        ops::dispatch_unary_inplace<ops::FillConst>(*this, value);
    }


    // -------------------------------------------------------------------------------------------------------------  
    //                                          GEOMETRIC OPERATIONS
    // -------------------------------------------------------------------------------------------------------------

    /* Checks if the strides represent a contiguous representation of data */
    bool TensorImpl::is_contiguous() const
    {
        // An empty tensor is considered as contiguous
        if (shape_.empty())
            return true;
        
        if (shape_.size() != strides_.size())
            return false;
        
        // Computes the strides backwards
        size_t expected_stride = 1;
        for (size_t i = shape_.size(); i-- > 0;) {
            
            // Tensor with a shape = 0 is empty -> contiguous
            if (shape_[i] == 0)
                return true;
            
            if (strides_[i] != expected_stride)
                return false;
            
            expected_stride *= shape_[i];
        }
        return true;
    }


    std::unique_ptr<TensorImpl> TensorImpl::view(std::vector<size_t>& new_shape) const
    {
        /**
         * TODO: add support for shape inferring. Currently only support the use of explicit shapes.
         * TODO: IMPLEMENTATION DOES NOT BELONG HERE, USE DISPATCHER.
         * TODO: add check for same shape (just copy the original)
         */

        if (!is_contiguous()) {
            throw std::runtime_error("view() called on a non-contiguous tensor. Call contiguous() first.");
        }

        size_t new_total = 1;
        for (size_t dim : new_shape) new_total *= dim;
        if (new_total != total_size_) {
            throw std::runtime_error("view(): shape is incompatible with the number of elements");
        }

        auto result = std::make_unique<TensorImpl>(*this);
        result->shape_ = new_shape;
        result->strides_ = {};
        result->refresh_metadata();

        result->autograd_meta_ = nullptr;
        if (requires_grad_) {
            // placeholder. Not sure how to handle this.
            std::make_unique<grad::AutogradMeta>();

            // This is going to be dropped unless backward_view is going to be supported.

            // In order to support that a ops/geometric.hpp file should be created.
            // And these operations are going to be executed by calling the dispatcher.
            
            // This OP has no prob with CUDA (no support of copy), but reshape does.
        }

        return result;
    }
    
    std::unique_ptr<TensorImpl> TensorImpl::reshape(std::initializer_list<size_t>& new_shape)
    {
        return std::make_unique<TensorImpl>(); //placeholder
    }

    std::unique_ptr<TensorImpl> TensorImpl::to_dtype(ScalarType target_dtype) const
    {
        if (this->dtype_ == target_dtype) {
            return std::make_unique<TensorImpl>(*this);
        }

        // TensorImpl result = ops::dispatch_unary_casting<ops::UnaryCastOp>(*this, target_dtype);
        // return std::make_shared<TensorImpl>(std::move(result));
        
        return std::make_unique<TensorImpl>();  // placeholder
    }

    // -------------------------------------------------------------------------------------------------------------  
    //                                          BINARY OPERATIONS
    // -------------------------------------------------------------------------------------------------------------

    TensorImpl add(TensorImpl& lhs, TensorImpl& rhs) {
        return ops::dispatch_binary<ops::BinaryAdd>(lhs, rhs);
    }


    /* Explicit instantiations */

    #define INSTANTIATE(T)                                                                              \
        template T* TensorImpl::data_ptr<T>();                                                          \
        template const T* TensorImpl::data_ptr<T>() const;                                              \
        
    #define INSTANTIATE_OP(T)                                                           \
        template T& TensorImpl::operator()<T>(const std::initializer_list<size_t>&);                    

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(int32_t)
    INSTANTIATE(int64_t)
    INSTANTIATE(bool)

    INSTANTIATE_OP(float)
    INSTANTIATE_OP(double)
    INSTANTIATE_OP(int32_t)
    INSTANTIATE_OP(int64_t)
    INSTANTIATE_OP(bool)

    #undef INSTANTIATE
    #undef INSTANTIATE_OP

} // namespace tensor
