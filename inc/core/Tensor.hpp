#pragma once
#include "Types.hpp"
#include "TensorImpl.hpp"
#include <vector>
#include <cstddef>
#include <memory>
#include <initializer_list>

namespace tensor
{
    
    class Node;
    
    // -------------------------------------------------------------------------------------------------------------  
    //                                                  TENSOR CLASS
    // ------------------------------------------------------------------------------------------------------------- 
    
    class Tensor
    {
    private:

        // ---------------------------- IMPLEMENTATION ---------------------------- 
        std::shared_ptr<TensorImpl> pimpl_;

        /* Create tensor from existing implementation*/
        Tensor(std::shared_ptr<TensorImpl> impl);


    
    public:
    
        // ------------------------------------------------------------------------------------------------------
        //                                           ACCESSORS 
        // ------------------------------------------------------------------------------------------------------ 
        
        // ---------------------------------- DATA MEMBERS ---------------------------------- 
        
        TensorImpl* get_impl() const;
        const std::vector<size_t>& get_shape() const;
        const std::vector<size_t>& get_strides() const;
        ScalarType get_dtype() const;
        Device get_device() const;
        size_t get_size() const;
        size_t get_dims() const;
        
        
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


        // ------------------------------------------------------------------------------------------------------
        //                                           CONSTRUCTORS 
        // ------------------------------------------------------------------------------------------------------ 
       
        // ---------------------------------- DEFAULT CONSTRUCTORS ---------------------------------- 
        
        /* Empty tensor: no TensorImpl member */
        Tensor();
        ~Tensor();
        Tensor(const Tensor& other);                    /* Shallow copy (view) */
        Tensor clone() const;                           /* Deep copy */


        // ---------------------------------- DEFAULT INITIALIZED CONSTRUCTORS ---------------------------------- 
        
        Tensor(const std::vector<size_t>& shape);
        Tensor(const std::vector<size_t>& shape, ScalarType dtype);
        Tensor(const std::vector<size_t>& shape, Device device);
        Tensor(const std::vector<size_t>& shape, ScalarType dtype, Device device);
        Tensor(ScalarType dtype, Device device);

        Tensor(std::initializer_list<int> shape);
        Tensor(std::initializer_list<int> shape, ScalarType dtype);
        Tensor(std::initializer_list<int> shape, Device device);
        Tensor(std::initializer_list<int> shape, ScalarType dtype, Device device);
        

        // ---------------------------------- CONST FILLING CONSTRUCTORS ---------------------------------- 

        Tensor(const std::vector<size_t>& shape, double fill_value);
        Tensor(const std::vector<size_t>& shape, ScalarType dtype, double fill_value);
        Tensor(const std::vector<size_t>& shape, Device device, double fill_value);
        Tensor(const std::vector<size_t>& shape, ScalarType dtype, Device device, double fill_value);
        Tensor(ScalarType dtype, Device device, double fill_value);

        Tensor(std::initializer_list<int> shape, double fill_value);
        Tensor(std::initializer_list<int> shape, ScalarType dtype, double fill_value);
        Tensor(std::initializer_list<int> shape, Device device, double fill_value);
        Tensor(std::initializer_list<int> shape, ScalarType dtype, Device device, double fill_value);



        // ------------------------------------------------------------------------------------------------------
        //                                           INDEXERS [], () 
        // ------------------------------------------------------------------------------------------------------ 
        // TODO: add const overload for all indexers
        
        template <typename T>
        T& operator[](size_t idx);

        template <typename T>
        T& operator()(const std::vector<size_t>& indices);

        template <typename T>
        T& operator()(const std::initializer_list<size_t>& indices);

        template <typename T>
        T& operator()(std::initializer_list<int> indices);

        template <typename T, typename... Args>
        T& operator()(Args... indices);


        // ------------------------------------------------------------------------------------------------------
        //                                           FILLING OPERATIONS
        // ------------------------------------------------------------------------------------------------------ 
        
        void fill_const(double value);
        void fill_arange(double start = 0.0, double step = 1.0);
        void fill_linspace(double start = 0.0, double end = 1.0);
        void fill_rand(double low = 0.0, double high = 1.0, uint64_t seed = 42);
        void fill_rand_normal(double mean = 0.0, double stddev = 1.0, uint64_t seed = 42);
        void fill_eye();

        // ------------------------------------------------------------------------------------------------------
        //                                           UNARY OPERATIONS
        // ------------------------------------------------------------------------------------------------------

        // ---------------------------------- GEOMETRIC OPERATIONS ---------------------------------- 
        
        Tensor contiguous() const;
        void contiguous_inplace();
        
        bool is_contiguous() const;
        
        Tensor view(std::vector<size_t>& shape) const;
        Tensor view(std::initializer_list<int> shape) const;
        
        Tensor reshape(std::vector<size_t>& shape) const;
        Tensor reshape(std::initializer_list<int> shape) const;
        

        // ---------------------------------- UNARY OPERATIONS ---------------------------------- 
        
        Tensor neg() const;
        Tensor abs() const;
        Tensor exp() const;
        Tensor log() const;
        
        Tensor sigmoid() const;
        Tensor tanh() const;
        Tensor relu() const;

        void neg_inplace();
        void abs_inplace();
        void exp_inplace();
        void log_inplace();
        
        void sigmoid_inplace();
        void tanh_inplace();
        void relu_inplace();


        // ------------------------------------------------------------------------------------------------------
        //                                           BINARY OPERATIONS
        // ------------------------------------------------------------------------------------------------------

        Tensor add(const Tensor& other) const;
        Tensor sub(const Tensor& other) const;
        Tensor exp(const Tensor& other) const;
        Tensor mul(const Tensor& other) const;
        Tensor div(const Tensor& other) const;
        
        void add_inplace(const Tensor& other);
        void sub_inplace(const Tensor& other);
        void exp_inplace(const Tensor& other);
        void mul_inplace(const Tensor& other);
        void div_inplace(const Tensor& other);
    };
    

    // ======================================================================================================
    //                                           FREE FUNCTIONS
    // ======================================================================================================

    // ---------------------------------- STATIC GENERATORS ---------------------------------- 
    
    Tensor zeros(const std::vector<size_t>& shape = {},
                 ScalarType dtype = ScalarType::Float32,
                 Device device = {DeviceType::CPU, 0});
    
    Tensor ones(const std::vector<size_t>& shape = {},
                ScalarType dtype = ScalarType::Float32,
                Device device = {DeviceType::CPU, 0});
    
    Tensor eye(size_t size = 0,
               ScalarType dtype = ScalarType::Float32,
               Device device = {DeviceType::CPU, 0});
    
    Tensor zeros(std::initializer_list<int> shape,
                 ScalarType dtype = ScalarType::Float32,
                 Device device = {DeviceType::CPU, 0});
            
    Tensor ones(std::initializer_list<int> shape,
                ScalarType dtype = ScalarType::Float32,
                Device device = {DeviceType::CPU, 0});
                
    
    // ---------------------------------- PRINTING UTILITIES ---------------------------------- 

    inline std::string metadata_to_string(const Tensor& tensor) {
        return metadata_to_string(*tensor.get_impl());
    }
    
    inline std::string to_string(const Tensor& tensor) {
        return to_string(*tensor.get_impl());
    }

    inline std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        return os << to_string(*tensor.get_impl());
    }

} // namespace tensor