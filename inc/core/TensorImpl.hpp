#pragma once 
#include "core/Storage.hpp"
#include "core/Types.hpp"
#include "core/Dispatchers.hpp"
#include "autograd/AutogradMeta.hpp"
#include <vector>
#include <cassert>
#include <memory>
#include <sstream>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace tensor
{

    // TODO: constructor to pass values to the storage (non-null initialization of vector)
    // TODO: check if view and clone are using same or new storage correctly

    struct TensorImpl
    {
        std::shared_ptr<Storage> storage_ = nullptr;
        ScalarType dtype_ =  ScalarType::Float32;
        Device device_ = {DeviceType::CPU, 0};
        size_t offset_ = 0;                         /* Elements offset */
        std::vector<size_t> shape_ = {};
        std::vector<size_t> strides_ = {};
        size_t total_size_ = 0;                     /* Number of elements */
        bool contiguous_ = true;

        bool requires_grad_ = false;
        std::unique_ptr<AutogradMeta> autograd_meta_ = nullptr;


        // -------------------------------------------------------------------------------------------------------------  
        //                                                  GETTERS
        // -------------------------------------------------------------------------------------------------------------        

        std::vector<size_t>& get_shape()   { return shape_; }
        std::vector<size_t>& get_strides() { return strides_; }
        ScalarType get_dtype()             { return dtype_; }
        size_t get_total_size()            { return total_size_; }
        Device get_device()                { return device_; }
        bool requires_grad()               { return requires_grad_; }
        bool get_contiguous()              { return contiguous_; }


        // -------------------------------------------------------------------------------------------------------------  
        //                                              HELPER FUNCTIONS
        // -------------------------------------------------------------------------------------------------------------

        // TODO: check the compute strides part, might need alternative computations for empty/non empty tensors
        /* Helper: computes metadata after chenges in view, shape, device, ... */
        void refresh_metadata()
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


        /* Helper: finds the actual memory offset for logical indexing (useful in element-wise operations)*/
        size_t get_physical_offset(const std::vector<size_t>& indices) const
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
        
        TensorImpl() = default;
        ~TensorImpl() = default;

        TensorImpl(const TensorImpl& other)
            : storage_(other.storage_), dtype_(other.dtype_), device_(other.device_), offset_(other.offset_),
              shape_(other.shape_), strides_(other.strides_), total_size_(other.total_size_),
              contiguous_(other.contiguous_), requires_grad_(other.requires_grad_) 
        {
            if (other.autograd_meta_)
                // TODO: might not work (check the AutograMeta struct) 
                autograd_meta_ = std::make_unique<AutogradMeta>(*other.autograd_meta_);
        }

        TensorImpl& operator=(const TensorImpl& other)
        {
            if (this != &other) {
                TensorImpl temp(other);
                std::swap(*this, temp);
            }
            return *this;
        }

        // TODO: Implementation must be in the .cpp file : requires complete types
        TensorImpl(TensorImpl&& other) noexcept; //=default
        TensorImpl& operator=(TensorImpl&& other) noexcept; //=default


        // ------------------------------------ CONSTRUCTOR OVERLOADS -------------------------------------

        TensorImpl(const std::vector<size_t>& shape,
                   ScalarType dtype = ScalarType::Float32,
                   Device device = {DeviceType::CPU, 0},
                   bool requires_grad = false)
            : shape_(std::move(shape)), dtype_(dtype), device_(std::move(device)), requires_grad_(requires_grad)
        {
            refresh_metadata();

            if(total_size_ > 0) {
                storage_ = std::make_shared<Storage>(total_size_ * element_size(dtype_), device_);
            } else {
                storage_ = nullptr;
            }

            if (requires_grad_) {
                autograd_meta_ = std::make_unique<AutogradMeta>();
            }
        }
        

        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        TensorImpl(const std::vector<size_t>& shape,
                   T fill_value,
                   ScalarType dtype = get_scalar_type<T>(),
                   Device device = {DeviceType::CPU, 0},
                   bool requires_grad = false)
            : TensorImpl(shape, dtype, device, requires_grad)
        {
            fill_const(fill_value);
        }
                   

        // TODO: requires check shape and type. Type should be from the 
        // storage? Storage handle this (requires some safety checks tho)
        TensorImpl(const std::vector<size_t>& shape,
                   ScalarType dtype = ScalarType::Float32,
                   Device device = {DeviceType::CPU, 0},
                   bool requires_grad = false,
                   void* src = nullptr);
        
        

        // -------------------------------------------------------------------------------------------------------------  
        //                                              INDEXERS/ACCESSORS
        // -------------------------------------------------------------------------------------------------------------

        // Data accessor helper
        template <typename T>
        T* data_ptr()
        {
            if (!storage_) return nullptr;
            return reinterpret_cast<T*>(static_cast<char*>(storage_->data()) + (offset_ * element_size(dtype_)));
        }

        void* data_ptr() {
            if (!storage_) return nullptr;
            return static_cast<char*>(storage_->data()) + (offset_ * element_size(dtype_));
        }

        const void* data_ptr() const
        {
            if (!storage_) return nullptr;
            return static_cast<const char*>(storage_->data()) + (offset_ * element_size(dtype_));
        }

        template <typename T>
        const T* data_ptr() const
        {
            return static_cast<const *T>(data_ptr());
        }

        // TODO: provide const version
        template <typename T>
        T& operator()(const std::initializer_list<size_t>& indices)
        {
            if (indices.size() != shape_.size())
                throw std::runtime_error("operator(): Index dimension mismatch");

            size_t idx = get_physical_offset(indices);
            return static_cast<T*>(storage_->data())[idx];
        }

        std::shared_ptr<TensorImpl> operator[](size_t index)
        {
            if (shape_.empty()) {
                throw std::runtime_error("operator[]: cannot index a 0-dim tensor.");
            }

            if (index >= shape_[0]) {
                throw std::out_of_range("Index out of range for dimension 0.");
            }


        }


        // -------------------------------------------------------------------------------------------------------------  
        //                                          FILLING OPERATIONS
        // -------------------------------------------------------------------------------------------------------------

        /**
         * Fill the tensor with a constant value. The provided value type is going to be
         * casted to the tensor ScalarType.
         */
        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        void fill_const(T value)
        {
            ops::dispatch_unary_inplace<ops::FillConst>(*this, value);
        }



        // -------------------------------------------------------------------------------------------------------------  
        //                                          GEOMETRIC OPERATIONS
        // -------------------------------------------------------------------------------------------------------------
    
        /* Checks if the strides represent a contiguous representation of data */
        bool is_contiguous() const
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


        std::unique_ptr<TensorImpl> view(std::vector<size_t>& new_shape) const;
        std::unique_ptr<TensorImpl> reshape(std::initializer_list<size_t>& new_shape);


        // Defined for the Iterator automatic casting. The type for the user operation might be different.
        // TODO: in this case. Move this to private and declare it as Iterator friend.
        std::unique_ptr<TensorImpl> to_dtype(ScalarType target_dtype) const
        {
            if (this->dtype_ == target_dtype) {
                return std::make_unique<TensorImpl>(*this);
            }

            // TensorImpl result = ops::dispatch_unary_casting<ops::UnaryCastOp>(*this, target_dtype);
            // return std::make_shared<TensorImpl>(std::move(result));
        }

    };


    // -------------------------------------------------------------------------------------------------------------  
    //                                         METADATA PRINTING UTITLIES
    // -------------------------------------------------------------------------------------------------------------

    inline std::string to_string(const TensorImpl& tensor)
    {
        std::ostringstream oss;

        oss << "TensorImpl {\n";

        if (tensor.storage_) {
            oss << "  " << to_string(*tensor.storage_) << std::endl;
        }
        else {
            oss << "  Storage: null" << std::endl;    
        }

        oss << "  DType: " << to_string(tensor.dtype_) << "\n";     // Device
        oss << "  Device: " << to_string(tensor.device_) << "\n";   // Device
        oss << "  Offset: " << tensor.offset_ << "\n";              // Offsets
        oss << "  Shape: [";                                        // Shape
        for (size_t i = 0; i < tensor.shape_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << tensor.shape_[i];
        }
        oss << "]\n";
        oss << "  Strides: [";                                      // Strides
        for (size_t i = 0; i < tensor.strides_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << tensor.strides_[i];
        }
        oss << "]\n";
        oss << "  Total Size: " << tensor.total_size_ << " elements\n";                     //size
        oss << "  Contiguous: " << (tensor.contiguous_ ? "true" : "false") << "\n";         // Contiguous flag
        oss << "  Requires Grad: " << (tensor.requires_grad_ ? "true" : "false") << "\n";   // Autograd information
        oss << "  Autograd Meta: " << (tensor.autograd_meta_ ? "present" : "null");

        oss << "\n}";
        
        return oss.str();
    }

    std::ostream& operator<<(std::ostream& os, const TensorImpl& tensor)
    {
        return os << to_string(tensor);
    }

} // namespace tensor