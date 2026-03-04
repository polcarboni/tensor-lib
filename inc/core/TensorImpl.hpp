#pragma once 
#include "core/Storage.hpp"
#include "core/Types.hpp"
#include "autograd/AutogradMeta.hpp"
#include <vector>
#include <memory>
#include <sstream>
#include <iomanip>
#include <functional>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace tensor
{

    // TODO: constructor to pass values to the storage (non-null initialization of vector)
    // TODO: check if view and clone are using same or new storage correctly
    struct Storage;
    struct Node;

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
        std::unique_ptr<grad::AutogradMeta> autograd_meta_ = nullptr;


        // -------------------------------------------------- GETTERS --------------------------------------------------

        std::vector<size_t>& get_shape();
        std::vector<size_t>& get_strides();
        ScalarType get_dtype();
        size_t get_total_size();
        Device get_device();
        bool requires_grad();
        bool get_contiguous();

        void set_shape(const std::vector<size_t>& shape);
        void set_strides(const std::vector<size_t>& strides);
        void set_dtype(ScalarType dtype);
        void set_total_size(size_t total_size);
        void set_device(const Device& device);
        void set_requires_grad(bool requires_grad);
        void set_contiguous(bool contiguous);


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
                   double fill_value,
                   ScalarType dtype = ScalarType::Float32,
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
        // template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        void fill_const(double value);


        // -------------------------------------------------- GEOMETRIC OPERATIONS --------------------------------------------------
        
        /* Checks if strides define a contiguous representation of data */
        bool is_contiguous() const;
        
        /**
         * Returns a new TensorImpl sharing the same underlying storage but a different shape.
         * The provided shape must have the same number of elements as the previous one.
         * 
         * Requires the TensorImpl to be contiguous.
         */
        std::unique_ptr<TensorImpl> view(std::vector<size_t>& new_shape) const;

        /**
         * Returns a new TensorImpl with the given shape, containing the same data.
         * If the tensor is contiguous the same storage is used, otherwise a contiguous copy is made first.
         */
        std::unique_ptr<TensorImpl> reshape(std::initializer_list<size_t>& new_shape);
        
        // Defined for the Iterator automatic casting. The type for the user operation might be different.
        // TODO: in this case. Move this to private and declare it as Iterator friend.
        std::unique_ptr<TensorImpl> to_dtype(ScalarType target_dtype) const;
        

        // -------------------------------------------------- AUTOGRAD METHODS --------------------------------------------------
        
        // Different metadata initialization if the tensor is a leaf or if it is generated by another operation
        void init_leaf_metadata();
        void init_intermediate_metadata(std::shared_ptr<Node> grad_fn);
        
        // Increment the version if an oepration modifies the storage data
        void bump_version();
        uint32_t get_version() const;

        // Return true if autograd_meta_ has no grad_fn_
        bool is_leaf() const;

        TensorImpl sum(const TensorImpl& lhs, const TensorImpl& rhs);
    };


    
    // -------------------------------------------- PRINTING UTITLIES --------------------------------------------

    inline std::string metadata_to_string(const TensorImpl& tensor)
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


    inline std::string to_string(const TensorImpl& tensor)
    {
        std::ostringstream oss;
        oss << std::setprecision(4) << std::fixed;

        const std::string prefix = "Tensor=(";
        const int prefix_len = static_cast<int>(prefix.size());
        oss << prefix;

        if (!tensor.storage_ || tensor.total_size_ == 0) {
            // TODO: add dtype, device anyway? I guess so (at least device)
            oss << "[])";
            return oss.str();
        }

        // ----------------------- CUDA support logic -----------------------

        std::vector<char> host_buffer;
        const void* data_ptr = nullptr;

        if (tensor.device_.type == DeviceType::CUDA) {
            size_t bytes = tensor.total_size_ * element_size(tensor.dtype_);
            host_buffer.resize(bytes);

            cudaError_t err = cudaMemcpy(host_buffer.data(), tensor.storage_->data(), bytes, cudaMemcpyDeviceToHost);

            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("to_string: CUDA memcpy failed: ") + cudaGetErrorString(err));
            }
            data_ptr = host_buffer.data();
        
        } else {
            data_ptr = tensor.storage_->data();
        }



        // Recursive lambda function
        std::function<void(size_t dim, std::vector<size_t>& indices, int indent)> print_dim;

        print_dim = [&](size_t dim, std::vector<size_t>& indices, int indent)
        {
            if (dim == tensor.shape_.size()) {
                size_t physical = tensor.get_physical_offset(indices);

                DISPATCH_ALL_TYPES(tensor.dtype_, "to_string", [&] {
                    oss << *reinterpret_cast<const scalar_t*>(static_cast<const char*>(data_ptr) + physical * element_size(tensor.dtype_));
                });

                return;
            }

            std::string pad(prefix_len + indent + 1, ' ');
            oss << "[";

            for (size_t i = 0; i < tensor.shape_[dim]; ++i) {
                indices[dim] = i;

                // Inner dimensions
                if (dim < tensor.shape_.size() - 1) {
                    if (i > 0) {
                        oss << ",\n";
                        if (tensor.shape_.size() - dim >= 4) oss << "\n";
                        oss << pad;
                    }
                    print_dim(dim + 1, indices, indent + 1);
                } else {
                    // Innermost dimension
                    if (i > 0) oss << ", ";
                    print_dim(dim + 1, indices, indent + 1);
                }
            }

            oss << "]";
        };

        std::vector<size_t> indices(tensor.shape_.size(), 0);
        print_dim(0, indices, 0);
        oss << ")";

        return oss.str();
    }

    inline std::ostream& operator<<(std::ostream& os, const TensorImpl& tensor)
    {
        return os << to_string(tensor);
    }

    TensorImpl add(TensorImpl& lhs, TensorImpl& rhs);

} // namespace tensor