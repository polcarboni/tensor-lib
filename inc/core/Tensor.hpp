#pragma once
#include "Types.hpp"
#include "TensorImpl.hpp"
#include <vector>
#include <cstddef>
#include <memory>

namespace tensor
{
    // -------------------------------------------------------------------------------------------------------------  
    //                                                  TENSOR CLASS
    // ------------------------------------------------------------------------------------------------------------- 

    class TensorImpl;
    class Node;

    /*
        TENSOR CLASS: wrapper of the TensorImpl class to be used as public API for the tensor library
        */

    class Tensor
    {
    private:

        // ---------------------------- IMPLEMENTATION ---------------------------- 
        std::unique_ptr<TensorImpl> pimpl_;
        Tensor(std::unique_ptr<TensorImpl> impl);   // Create tensor from existing implementation
    
    public:

        // ---------------------------------- ACCESSORS ---------------------------------- 
        
        TensorImpl* impl() const;
        const std::vector<size_t>& shape() const;
        const std::vector<size_t>& strides() const;
        ScalarType dtype() const;
        Device device() const;
        size_t size() const;
        size_t dims() const;
        
        
        // ---------------------------------- GRAD ACCESSORS ---------------------------------- 
        
        bool requires_grad() const;
        void set_requires_grad(bool r);

        void backward(const Tensor& gradient, bool retain_graph = false, bool create_graph = false);
        Tensor grad() const;
        void set_grad(Tensor grad);

        std::shared_ptr<Node> grad_fn() const;
        void set_grad_fn(std::shared_ptr<Node> fn);

        bool is_leaf() const;
        uint32_t output_nr() const;

        // ---------------------------------- CONSTRUCTORS ---------------------------------- 
        
        // ---------------------------------- empty constructors ---------------------------------- 
        
        Tensor();   /* Empty tensor: no TensorImpl member */
        Tensor(const std::vector<size_t>& shape);
        Tensor(const std::vector<size_t>& shape, ScalarType dtype = ScalarType::Float32, Device device = {DeviceType::CPU, 0});
        ~Tensor();
        Tensor(const Tensor& other);    /* Shallow copy (view) */
        Tensor clone() const;   /* Deep copy */

        // ---------------------------------- overloaded (?) constructors ---------------------------------- 

        /* Construct tensor from: shape (std::vector<size_t>), values(std::vector<T>) and device */
        // template<typename T>
        // Tensor(const std::vector<size_t>& shape, const std::vector<T>& values, Device device);

        // ---------------------------------- UTILITY FUNCTIONS ---------------------------------- 

        Tensor contiguous() const;
        bool is_contiguous() const;
        Tensor view(std::vector<size_t>& shape) const;
        Tensor reshape(std::vector<size_t>& shape) const;
    };
    
} // namespace tensor