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

        /* Create tensor from existing implementation*/
        Tensor(std::unique_ptr<TensorImpl> impl)
            : pimpl_(std::move(impl)) { }
    
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
        
        /* Empty tensor: no TensorImpl member */
        Tensor() : Tensor(std::make_unique<TensorImpl>()) {}

        ~Tensor();
        Tensor(const Tensor& other);                    /* Shallow copy (view) */
        Tensor clone() const;                           /* Deep copy */


        // ---------------------------------- overloaded (?) constructors ---------------------------------- 

        /* Construct tensor from: shape (std::vector<size_t>), values(std::vector<T>) and device */
        // template<typename T>
        // Tensor(const std::vector<size_t>& shape, const std::vector<T>& values, Device device);

        
        
        // Tensor(const std::vector<size_t>& shape,
        //        ScalarType dtype = ScalarType::Float32,
        //        Device device = {DeviceType::CPU, 0})
        //     : Tensor(std::make_unique<TensorImpl>(shape, dtype, device)) { }


        // TODO: TensorImpl.cpp and Storage.cpp, add constructors that take as input a value for filling
        // TODO: overload constructors with std::vector<size_t>

        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        Tensor(const std::initializer_list<size_t>& shape, const T fill_value);

        static Tensor zeros(const std::initializer_list<size_t>& shape,
                            ScalarType dtype = ScalarType::Float32,
                            Device device = {DeviceType::CPU, 0})
        {
            auto impl = std::make_unique<TensorImpl>(shape, dtype, device, 0.0);
            return Tensor(std::move(impl));
        }

        static Tensor ones(const std::initializer_list<size_t>& shape,
                            ScalarType dtype = ScalarType::Float32,
                            Device device = {DeviceType::CPU, 0})
        {
            auto impl = std::make_unique<TensorImpl>(shape, dtype, device, 1.0);
            return Tensor(std::move(impl));
        }

        static Tensor eye(size_t size,
                          ScalarType dtype = ScalarType::Float32,
                          Device device = {DeviceType::CPU, 0});
        // {
            // // TODO-fix: {size, size} not an initializer list
            // auto impl = std::make_unique<TensorImpl>({size, size}, dtype, device, 1.0);
            // return Tensor(std::move(impl));
        // }


        // ------------------------------------------------------------------------------------------------------
        //                                           INDEXERS [], () 
        // ------------------------------------------------------------------------------------------------------ 
        
        // TODO: add const overload for all indexers
        
        template <typename T>
        T& operator[](size_t idx)
        {

        }

        template <typename T>
        T& at(size_t idx)
        {
            // Bounded access to element
        }

        template <typename T>
        T& operator()(const std::initializer_list<size_t>& indices)
        {

        }

        template <typename T, typename... Args>
        T& operator()(Args... dims)
        {

        }


        // ---------------------------------- UTILITY FUNCTIONS ---------------------------------- 

        Tensor contiguous() const;
        bool is_contiguous() const;
        Tensor view(std::vector<size_t>& shape) const;
        Tensor reshape(std::vector<size_t>& shape) const;
    };
    
} // namespace tensor