#pragma once 
#include "core/Storage.hpp"
#include "core/Types.hpp"
#include "autograd/AutogradMeta.hpp"
#include <vector>
#include <memory>
#include <sstream>

namespace tensor
{

    // TODO: constructor to pass values to the storage (non-null initialization of vector)
    // TODO: check if view and clone are using same or new storage correctly
    struct Storage;

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


        // -------------------------------------------------- GETTERS --------------------------------------------------

        std::vector<size_t>& get_shape();
        std::vector<size_t>& get_strides();
        ScalarType get_dtype();
        size_t get_total_size();
        Device get_device();
        bool requires_grad();
        bool get_contiguous();


        // -------------------------------------------------- HELPER FUNCTIONS --------------------------------------------------

        // TODO: check the compute strides part, might need alternative computations for empty/non empty tensors
        /* Helper: computes metadata after chenges in view, shape, device, ... */
        void refresh_metadata();

        /* Helper: finds the actual memory offset for logical indexing (useful in element-wise operations)*/
        size_t get_physical_offset(const std::vector<size_t>& indices) const;
    

        // -------------------------------------------------- CONSTRUCTORS --------------------------------------------------
        
        TensorImpl();
        ~TensorImpl();
        TensorImpl(const TensorImpl& other);
        TensorImpl& operator=(const TensorImpl& other);
        TensorImpl(TensorImpl&& other) noexcept;
        TensorImpl& operator=(TensorImpl&& other) noexcept;

        TensorImpl(const std::vector<size_t>& shape,
                   ScalarType dtype = ScalarType::Float32,
                   Device device = {DeviceType::CPU, 0},
                   bool requires_grad = false);        

        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        TensorImpl(const std::vector<size_t>& shape,
                   T fill_value,
                   ScalarType dtype = get_scalar_type<T>(),
                   Device device = {DeviceType::CPU, 0},
                   bool requires_grad = false);

        // TODO: requires check shape and type. Type should be from the 
        // storage? Storage handle this (requires some safety checks tho)
        TensorImpl(const std::vector<size_t>& shape,
                   ScalarType dtype = ScalarType::Float32,
                   Device device = {DeviceType::CPU, 0},
                   bool requires_grad = false,
                   void* src = nullptr);
        
        

        // -------------------------------------------------- INDEXERS/ACCESSORS --------------------------------------------------

        // Data accessor helper
        template <typename T>
        T* data_ptr();

        void* data_ptr();

        const void* data_ptr() const;

        template <typename T>
        const T* data_ptr() const;

        // TODO: provide const version
        template <typename T>
        T& operator()(const std::initializer_list<size_t>& indices);

        std::shared_ptr<TensorImpl> operator[](size_t index);


        // -------------------------------------------------- FILLING OPERATIONS --------------------------------------------------

        /**
         * Fill the tensor with a constant value. The provided value type is going to be
         * casted to the tensor ScalarType.
         */
        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        void fill_const(T value);


        // -------------------------------------------------- GEOMETRIC OPERATIONS --------------------------------------------------
    
        /* Checks if the strides represent a contiguous representation of data */
        bool is_contiguous() const;

        std::unique_ptr<TensorImpl> view(std::vector<size_t>& new_shape) const;
        std::unique_ptr<TensorImpl> reshape(std::initializer_list<size_t>& new_shape);

        // Defined for the Iterator automatic casting. The type for the user operation might be different.
        // TODO: in this case. Move this to private and declare it as Iterator friend.
        std::unique_ptr<TensorImpl> to_dtype(ScalarType target_dtype) const;

    };


    
    // -------------------------------------------- METADATA PRINTING UTITLIES --------------------------------------------

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

    inline std::ostream& operator<<(std::ostream& os, const TensorImpl& tensor)
    {
        return os << to_string(tensor);
    }

} // namespace tensor