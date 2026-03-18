#include "core/Types.hpp"
#include "core/Tensor.hpp"
#include "core/TensorImpl.hpp"
#include <stdexcept>
#include <vector>
#include <memory>

/**
 * TODO: check if the use of return is correct in void inplace operations with
 *       API call to impl_ function.
 */


namespace {

    /**
     * Utility namespace for the conversion of the API initializer_lists in std::vector<size_t>
     * compatible with the TensorImpl class.
     * 
     * Adds a check layer on the provided values.
     */

    std::vector<size_t> checked_shape(std::initializer_list<int> il) {
        std::vector<size_t> v;
        v.reserve(il.size());
        for (int x : il) {
            if (x < 0) {
                throw std::invalid_argument("negative dimensions not allowed.");
            }
            v.push_back(static_cast<size_t>(x));
        }
        return v;
    }

    std::vector<size_t> checked_indices(std::initializer_list<int> il) {
        std::vector<size_t> v;
        v.reserve(il.size());
        for (int x : il) {
            if (x < 0) {
                throw std::invalid_argument("negative indices not allowed.");
            }
            v.push_back(static_cast<size_t>(x));
        }
        return v;
    }
}


namespace tensor
{

    // ------------------------------------------------------------------------------------------------------
    //                                           ACCESSORS 
    // ------------------------------------------------------------------------------------------------------

    // ---------------------------------- DATA MEMBERS ---------------------------------- 

    TensorImpl* Tensor::get_impl() const {
        return pimpl_.get();
    }

    const std::vector<size_t>& Tensor::get_shape() const {
        return pimpl_->get_shape();
    }

    const std::vector<size_t>& Tensor::get_strides() const {
        return pimpl_->get_strides();
    }

    ScalarType Tensor::get_dtype() const {
        return pimpl_->get_dtype();
    }

    Device Tensor::get_device() const {
        return pimpl_->get_device();
    }

    size_t Tensor::get_size() const {
        return pimpl_->get_total_size();
    }

    size_t Tensor::get_dims() const {
        return pimpl_->get_shape().size();
    }

    // ---------------------------------- GRAD ACCESSORS ---------------------------------- 

    // TODO: not implemented yet


    // ------------------------------------------------------------------------------------------------------
    //                                           CONSTRUCTORS 
    // ------------------------------------------------------------------------------------------------------ 
    
    // ---------------------------------- PRIVATE CONSTRUCTORS ---------------------------------- 

    Tensor::Tensor(std::shared_ptr<TensorImpl> impl)
        : pimpl_(std::move(impl)) {}
        

    // ---------------------------------- DEFAULT CONSTRUCTORS ---------------------------------- 

    Tensor::Tensor()
        : pimpl_(nullptr) {}
    
    Tensor::~Tensor() = default;
    
    Tensor::Tensor(const Tensor& other)
        : pimpl_(other.pimpl_) { }

    Tensor Tensor::clone() const
    {
        return Tensor(); // placeholder
    }


    
    // ---------------------------------- DEFAULT INITIALIZED CONSTRUCTORS ---------------------------------- 

    Tensor::Tensor(const std::vector<size_t>& shape)
        : pimpl_(std::make_shared<TensorImpl>(shape)) {}

    Tensor::Tensor(const std::vector<size_t>& shape, ScalarType dtype)
        : pimpl_(std::make_shared<TensorImpl>(shape, dtype)) {}

    Tensor::Tensor(const std::vector<size_t>& shape, Device device)
        : pimpl_(std::make_shared<TensorImpl>(shape, ScalarType::Float32, device)) {}
    
    Tensor::Tensor(const std::vector<size_t>& shape, ScalarType dtype, Device device)
        : pimpl_(std::make_shared<TensorImpl>(shape, dtype, device)) {}

    Tensor::Tensor(ScalarType dtype, Device device)
        : pimpl_(std::make_shared<TensorImpl>(std::vector<size_t>{}, dtype, device)) {}
    

    Tensor::Tensor(std::initializer_list<int> shape)
        : Tensor(checked_shape(shape)) {}

    Tensor::Tensor(std::initializer_list<int> shape, ScalarType dtype)
        : Tensor(checked_shape(shape), dtype) {}

    Tensor::Tensor(std::initializer_list<int> shape, Device device)
        : Tensor(checked_shape(shape), device) {}

    Tensor::Tensor(std::initializer_list<int> shape, ScalarType dtype, Device device)
        : Tensor(checked_shape(shape), dtype, device) {}

        
    // ---------------------------------- CONST FILLING CONSTRUCTORS ---------------------------------- 

    Tensor::Tensor(const std::vector<size_t>& shape, double fill_value)
        : Tensor(shape) { fill_const(fill_value); }

    Tensor::Tensor(const std::vector<size_t>& shape, ScalarType dtype, double fill_value)
        : Tensor(shape, dtype) { fill_const(fill_value); }
        
    Tensor::Tensor(const std::vector<size_t>& shape, Device device, double fill_value)
        : Tensor(shape, ScalarType::Float32, device) { fill_const(fill_value); }
    
    Tensor::Tensor(const std::vector<size_t>& shape, ScalarType dtype, Device device, double fill_value)
        : Tensor(shape, dtype, device) { fill_const(fill_value); }
    
    Tensor::Tensor(ScalarType dtype, Device device, double fill_value)
        : Tensor(std::vector<size_t>{}, dtype, device) { fill_const(fill_value); }

    
    Tensor::Tensor(std::initializer_list<int> shape, double fill_value)
        : Tensor(checked_shape(shape), fill_value) {}

    Tensor::Tensor(std::initializer_list<int> shape, ScalarType dtype, double fill_value)
        : Tensor(checked_shape(shape), dtype, fill_value) {}

    Tensor::Tensor(std::initializer_list<int> shape, Device device, double fill_value)
        : Tensor(checked_shape(shape), device, fill_value) {}

    Tensor::Tensor(std::initializer_list<int> shape, ScalarType dtype, Device device, double fill_value)
        : Tensor(checked_shape(shape), dtype, device, fill_value) {}



    // ------------------------------------------------------------------------------------------------------
    //                                           INDEXERS [], () 
    // ------------------------------------------------------------------------------------------------------ 
    
    template <typename T>
    T& Tensor::operator[](size_t idx) {
        // TODO: implement
        throw std::logic_error("Tensor::operator[] not implemented");
    }    
    
    template <typename T>
    T& Tensor::operator()(const std::vector<size_t>& indices)
    {
        return pimpl_->operator()<T>(indices);
    }
    
    template <typename T>
    T& Tensor::operator()(const std::initializer_list<size_t>& indices) {
        return pimpl_->operator()<T>(indices);
    }

    template <typename T>
    T& Tensor::operator()(std::initializer_list<int> indices) {
        return operator()<T>(checked_indices(indices));
    }

    template <typename T, typename... Args>
    T& Tensor::operator()(Args... indices)
    {
        std::vector<size_t> idx = { static_cast<size_t>(indices)... };
        return pimpl_->operator()<T>(idx);
    }

    // ------------------------------------------------------------------------------------------------------
    //                                           FILLING OPERATIONS
    // ------------------------------------------------------------------------------------------------------ 

    void Tensor::fill_const(double value) { pimpl_->fill_const(value); }
    void Tensor::fill_arange(double start, double step) { pimpl_->fill_arange(start, step); }
    void Tensor::fill_linspace(double start, double end) { pimpl_->fill_linspace(start, end); }
    void Tensor::fill_rand(double low, double high, uint64_t seed) { pimpl_->fill_rand(low, high, seed); }
    void Tensor::fill_rand_normal(double mean, double stddev, uint64_t seed) { pimpl_->fill_rand_normal(mean, stddev, seed); }
    void Tensor::fill_eye() { pimpl_->fill_eye(); }


    // ------------------------------------------------------------------------------------------------------
    //                                           UNARY OPERATIONS
    // ------------------------------------------------------------------------------------------------------
    
    // ---------------------------------- GEOMETRIC OPERATIONS ---------------------------------- 

    Tensor Tensor::contiguous() const
    {
        Tensor result;
        result.pimpl_ = std::make_shared<TensorImpl>(pimpl_->contiguous());
        return result;
    }

    void Tensor::contiguous_inplace() {
        return pimpl_->contiguous_inplace();
    }

    bool Tensor::is_contiguous() const {
        return pimpl_->is_contiguous();
    }

    Tensor Tensor::view(std::vector<size_t>& shape) const
    {
        Tensor result;
        result.pimpl_ = std::make_shared<TensorImpl>(pimpl_->view(shape));
        return result;
    }

    Tensor Tensor::view(std::initializer_list<int> shape) const
    {
        std::vector<size_t> s = checked_shape(shape);
        return view(s);
    }

    Tensor Tensor::reshape(std::vector<size_t>& shape) const
    {
        Tensor result;
        result.pimpl_ = std::make_shared<TensorImpl>(pimpl_->reshape(shape));
        return result;
    }

    Tensor Tensor::reshape(std::initializer_list<int> shape) const
    {
        std::vector<size_t> s = checked_shape(shape);
        return reshape(s);
    }

    // ---------------------------------- UNARY OPERATIONS ---------------------------------- 

    Tensor Tensor::neg() const {
        Tensor result;
        result.pimpl_ = std::make_shared<TensorImpl>(pimpl_->neg());
        return result;
    }

    Tensor Tensor::abs() const {
        Tensor result;
        result.pimpl_ = std::make_shared<TensorImpl>(pimpl_->abs());
        return result;
    }

    Tensor Tensor::exp() const {
        Tensor result;
        result.pimpl_ = std::make_shared<TensorImpl>(pimpl_->exp());
        return result;
    }
    
    Tensor Tensor::log() const {
        Tensor result;
        result.pimpl_ = std::make_shared<TensorImpl>(pimpl_->log());
        return result;
    }
    
    Tensor Tensor::sigmoid() const {
        Tensor result;
        result.pimpl_ = std::make_shared<TensorImpl>(pimpl_->sigmoid());
        return result;
    }
    
    Tensor Tensor::tanh() const {
        Tensor result;
        result.pimpl_ = std::make_shared<TensorImpl>(pimpl_->tanh());
        return result;
    }
    
    Tensor Tensor::relu() const {
        Tensor result;
        result.pimpl_ = std::make_shared<TensorImpl>(pimpl_->relu());
        return result;
    }

    void Tensor::neg_inplace() { return pimpl_->neg_inplace(); }
    void Tensor::abs_inplace() { return pimpl_->abs_inplace(); }
    void Tensor::exp_inplace() { return pimpl_->exp_inplace(); }
    void Tensor::log_inplace() { return pimpl_->log_inplace(); }
    void Tensor::sigmoid_inplace() { return pimpl_->sigmoid_inplace(); }
    void Tensor::tanh_inplace() { return pimpl_->tanh_inplace(); }
    void Tensor::relu_inplace() { return pimpl_->relu_inplace(); }
    
    
    // ------------------------------------------------------------------------------------------------------
    //                                           BINARY OPERATIONS
    // ------------------------------------------------------------------------------------------------------
    
    // TO BE IMPLEMENTED



    // ======================================================================================================
    //                                           FREE FUNCTIONS
    // ======================================================================================================

    // ---------------------------------- STATIC GENERATORS ---------------------------------- 

    Tensor zeros(const std::vector<size_t>& shape,
                 ScalarType dtype,
                 Device device)
    {
        return Tensor(shape, dtype, device, 0.0);
    }
    
    Tensor ones(const std::vector<size_t>& shape,
                ScalarType dtype,
                Device device)
    {
        return Tensor(shape, dtype, device, 1.0);
    }
    
    Tensor eye(size_t size,
               ScalarType dtype,
               Device device)
    {
        Tensor tensor(std::vector<size_t>{size,size}, dtype, device, 0.0);
        tensor.fill_eye();
        return tensor;
    }


    Tensor zeros(std::initializer_list<int> shape, ScalarType dtype, Device device)
    {
        return zeros(checked_shape(shape), dtype, device);
    }

    Tensor ones(std::initializer_list<int> shape, ScalarType dtype, Device device)
    {
        return ones(checked_shape(shape), dtype, device);
    }


    




    /* Explicit instantiations */

    #define INSTANTIATE_TENSOR_OP(T)                                                                        \
        /* vector<size_t> */                                                                                \
        template T& Tensor::operator()<T>(const std::vector<size_t>&);                                     \
                                                                                                            \
        /* initializer_list<size_t> */                                                                      \
        template T& Tensor::operator()<T>(const std::initializer_list<size_t>&);                           \
                                                                                                            \
        /* initializer_list<int> */                                                                         \
        template T& Tensor::operator()<T>(std::initializer_list<int>);                                     \
                                                                                                            \
        template T& Tensor::operator[]<T>(size_t);                                                          \
                                                                                                            \
        /* variadic size_t: 1-D through 4-D */                                                             \
        template T& Tensor::operator()<T, size_t>(size_t);                                                 \
        template T& Tensor::operator()<T, size_t, size_t>(size_t, size_t);                                 \
        template T& Tensor::operator()<T, size_t, size_t, size_t>(size_t, size_t, size_t);                 \
        template T& Tensor::operator()<T, size_t, size_t, size_t, size_t>(size_t, size_t, size_t, size_t); \
                                                                                                            \
        /* variadic int: 1-D through 4-D */                                                                \
        template T& Tensor::operator()<T, int>(int);                                                       \
        template T& Tensor::operator()<T, int, int>(int, int);                                             \
        template T& Tensor::operator()<T, int, int, int>(int, int, int);                                   \
        template T& Tensor::operator()<T, int, int, int, int>(int, int, int, int);
    
    INSTANTIATE_TENSOR_OP(float)
    INSTANTIATE_TENSOR_OP(double)
    INSTANTIATE_TENSOR_OP(int32_t)
    INSTANTIATE_TENSOR_OP(int64_t)
    INSTANTIATE_TENSOR_OP(bool)

    #undef INSTANTIATE_TENSOR_OP
} // namespace tensor
