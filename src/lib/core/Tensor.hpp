#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <initializer_list>
#include <numeric>
#include <algorithm>
#include <type_traits>

namespace tensor {

    template<typename T>
    class Tensor
    {
        static_assert(std::is_arithmetic<T>::value, "Tensor class requires a numeric type (integral or floating-point)");

    private:
        std::shared_ptr<T[]> data_;             // Elements contained in the tensor
        std::vector<size_t> shape_;             // Tensor shape
        std::vector<size_t> strides_;           // Tensor strides for element access
        size_t total_size_;                     // Total numbers of elements in the tensor
        
        //TODO: adding data members requires updating the functions 
        // size_t offset = 0;              // CAN BE USEFUL BUT FOR WHAT?

        //Computes strides and total size from the shape
        void compute_strides()
        {
            const size_t dims = shape_.size();
            strides_.resize(dims);
            
            if (dims == 0) {
                strides_= {};
                total_size_ = 1;
                return;
            }

            std::exclusive_scan(
                shape_.rbegin(), shape_.rend(),  // input range
                strides_.rbegin(),               // output range
                size_t(1),
                std::multiplies<size_t>()
            );

            total_size_ = strides_[0] * shape_[0];
        }

    public:

        // ------------------------------------------------------------------------------------------------------
        //                                              CONSTRUCTORS 
        // ------------------------------------------------------------------------------------------------------ 
        
        // Constructor: Empty Tensor 
        Tensor() : data_(nullptr), total_size_(0) {}

        ~Tensor() = default;

        // Copy constructor
        Tensor(const Tensor& other)
            : shape_(other.shape_), strides_(other.strides_), total_size_(other.total_size_)
        {
            if (total_size_ > 0)
            {
                data_ = std::shared_ptr<T[]>(new T[total_size_]);
                std::copy(other.data_.get(), other.data_.get() + total_size_, data_.get());
            }
        }

        // Move constructor
        Tensor(Tensor&& other) noexcept
            : data_(std::move(other.data_))
            , shape_(std::move(other.shape_))
            , strides_(std::move(other.strides_))
            , total_size_(other.total_size_)
        {
            other.total_size_ = 0;
        }

        // Constructor from shape and value: Tensor({x,y}, val)
        Tensor(std::vector<size_t> shape, T val = T{})
            : shape_(std::move(shape))
        {
            compute_strides();

            if (total_size_ > 0) {
                data_ = std::shared_ptr<T[]>(new T[total_size_]);
                std::fill(data_.get(), data_.get() + total_size_, val);
            } else {
                data_ = nullptr;
            }
        }

        // Constructor from initializer list: Tensor ({2,2}, {0,1,2,3})
        

        Tensor(std::vector<size_t> shape, std::initializer_list<T> list)
            : shape_(std::move(shape))
        {
            compute_strides();
            
            // ERROR MESSAGE: shape and initializer list mismatch
            if (list.size() != total_size_) {
                std::stringstream ss;
                ss << "ERROR: Data size [" << list.size()
                << "] does not match the dimension [" << total_size_
                << "] of the shape [";
                
                for (size_t i = 0; i < shape_.size(); ++i)
                {
                    ss << shape_[i] << (i == shape_.size() -1 ? "" : ",");
                }
                ss << "]";
                throw std::runtime_error(ss.str());
            }

            if (total_size_ > 0)
            {
                data_ = std::shared_ptr<T[]>(new T[total_size_]);
                std::copy(list.begin(), list.end(), data_.get());
            } else {
                data_ = nullptr;
            }
        }

        // TODO: constructor from 'nested initializer list' without specification of the shape 
        // {{0,1},{2,3}}. Requires recursive function
        
        // TODO: constructor from vector. (a vector of shapes and a vector of values)
        // TODO: static generators like Tensor::zeros(shape), Tensor::one(shape), Tensor::eye(shape)
        // TODO: casting ?


        // ------------------------------------------------------------------------------------------------------
        //                                 STATIC GENERATORS (ZEROS, ONES, EYE) 
        // ------------------------------------------------------------------------------------------------------ 
        

        // ----------------- TENSOR::ZEROS() ----------------- 

        // Tensor::zeros({1,2}): zeros from shape vector
        static Tensor zeros(const std::vector<size_t>& shape) {
            return Tensor(shape, T{});
        }

        // Tensor::zeros(5): one-dimensional tensor
        static Tensor zeros(size_t size)
        {
            return Tensor(std::vector<size_t>{size}, T{});
        }
        
        //Tensor::zeros(2,3,4): using variadic arguments (no vector)
        template <typename... Args>
        static Tensor zeros(Args... dims)
        {
            return Tensor(std::vector<size_t>{static_cast<size_t>(dims)...}, T{});
        }


        // ----------------- TENSOR::ONES() ----------------- 

        static Tensor ones(const std::vector<size_t>& shape)
        {
            return Tensor(shape, T{1});
        }

        static Tensor ones(size_t size)
        {
            return Tensor(std::vector<size_t>{size}, T{1});
        }

        template<typename... Args>
        static Tensor ones(Args... dims)
        {
            return Tensor(std::vector<size_t>{static_cast<size_t>(dims)...}, T{1});
        }

        // ----------------- TENSOR::EYE() -----------------

        static Tensor eye(const size_t n)
        {
            Tensor result({n,n}, T{0});
            for (size_t i = 0; i < n; ++i)
            {
                result(i,i) = T{1};
            }
            return result;
        }

        //TODO (why): rectangular identity matrix from shape (x,y) and from {x,y}

        // ----------------- GETTERS ----------------- 

        const std::vector<size_t>& shape() const { return shape_; };
        const std::vector<size_t>& strides() const { return strides_; };
        size_t size() const { return total_size_; };
        T* data() { return data_.get(); }
        const T* data() const { return data_.get(); }
        size_t ndim() const { return shape_.size(); }


        // ------------------------------------------------------------------------------------------------------
        //                                           INDEXERS [], () 
        // ------------------------------------------------------------------------------------------------------ 

        // ----------------- INDEXING [] -----------------
        // Access elements with a flat index

        T& operator[](size_t idx)
        {
            return data_[idx];
        }

        const T& operator[](size_t idx) const
        {
            return data_[idx];
        }

        T& at(size_t idx)
        {
            if (idx >= total_size_) {
                throw std::out_of_range("Index out of bounds");
            }
            return data_[idx];
        }

        const T& at(size_t idx) const
        {
            if (idx >= total_size_) {
                throw std::out_of_range("Index out of bounds");
            }
            return data_[idx];
        }
        

        // ----------------- INDEXING () ----------------- 

        // Indexing with () operator: Tensor({1,0,2}) that maps multdimensional index to the 
        // flattened version of the vector

        //TODO-fix: define the calculate flat index as a private method

        T& operator()(const std::vector<size_t>& indices) {
            //TODO: add error on the length of the list
            
            if (indices.size() != shape_.size()) 
            {
                throw std::runtime_error("ERROR: the provided list of indices does not match the shape.");
            }

            size_t flat_idx = 0;
            
            for (size_t i = 0; i < indices.size(); i++) {
                flat_idx += indices[i] * strides_[i];
            }
            return data_[flat_idx];
        }

        const T& operator()(const std::vector<size_t>& indices) const {
            //TODO: add error on the length of the list
            size_t flat_idx = 0;
            for (size_t i = 0; i < indices.size(); i++) {
                flat_idx += indices[i] * strides_[i];
            }
            return data_[flat_idx];
        }


        // Indexing: Tensor(3,2)
        template<typename... Args>
        T& operator()(Args... dims)
        {
            static_assert(sizeof...(Args) > 0, "Must provide indices.");
            size_t indices[] = { static_cast<size_t>(dims)... };

            if (sizeof...(Args) != shape_.size())
            {
                throw std::runtime_error("Index dimension mismatch");
            }

            size_t flat_idx = 0;
            for (size_t i = 0; i < sizeof...(Args); ++i)
            {
                flat_idx += indices[i] * strides_[i];
            }
            return data_[flat_idx];
        }

        // ------------------------------------------------------------------------------------------------------
        //                                            OPERATORS =, ==
        // ------------------------------------------------------------------------------------------------------ 

        // ----------------- ASSIGNMENT OPERATORS = ----------------- 

        Tensor& operator=(const Tensor& other)
        {
            if (this != &other)
            {
                shape_ = other.shape_;
                strides_ = other.strides_;
                total_size_ = other.total_size_;

                if (total_size_ > 0)
                {
                    data_ = std::shared_ptr<T[]>(new T[total_size_]);
                    std::copy(other.data_.get(), other.data_.get() + total_size_, data_.get());
                } else {
                    data_ = nullptr;
                }
            }
            return *this;
        }

        Tensor& operator=(Tensor&& other) noexcept
        {
            if (this != &other) {
                data_       = std::move(other.data_);
                shape_      = std::move(other.shape_);
                strides_    = std::move(other.strides_);
                total_size_ = std::move(other.total_size_);

                other.total_size_ = 0;
            }
            return *this;
        }

        // ----------------- COMPARISON OPERATOR == ----------------- 
        
        bool operator==(const Tensor& other) const
        {
            //TODO-fix: floating point tolerance
            //TODO-fix: check what to do for non-contiguous tensors
            if (total_size_ != other.total_size_ || shape_.size() != other.shape_.size()) {
                return false;
            }

            if (shape_ != other.shape_) {
                return false;
            }

            // Check if the two tensors are copies
            if (data_ == other.data_) {
                return true;
            }

            // Empty tensors pointing to separate addresses
            if (!data_ || !other.data_) {
                return total_size_ == 0;
            }

            return std::equal(data_.get(), data_.get() + total_size_,
                              other.data_.get());
        }

        bool operator!=(const Tensor<T>& other) const
        {
            return !(*this == other);
        }



        
        // TODO: check the contiguity of the tensor before implementing view
        //        (might be broken by some other member function)
        
        // Tensor view(const std::vector<size_t>& new_shape) const
        // {
        //     //TODO: add checking the size
        //     if (shape_length(new_shape) != shape_length(shape_))
        //     {
        //         throw std::runtime_error()
        //     }
        //     Tensor out;
        //     out.data_ = data_;
        //     out.shape_ = new_shape;
        //     out.compute_strides();
            
        //     return out;
        // }
    
        // TODO: transpose

        // ----------------- TRANSPOSE ----------------- 

        Tensor transpose(std::vector<size_t> axes = {}) const
        {
            const size_t dims = shape_.size();
        }
        // TODO: permute
        // TODO: slicing
        // TODO: views (understand what this means first)
        // TODO: reshape, flatten
        // TODO: broadcasting

        // TODO: HELPER FOR PRINTING: define methods that convert stuff to strings
        // in order to have better functionalities.




        // ------------------------------------------------------------------------------------------------------
        //                                            PRINT/GRAPHICAL
        // ------------------------------------------------------------------------------------------------------ 
        
        // ----------------- PRINT/VISUAL ----------------- 

        // TODO-fix: flat print(), modify for a better result
        void print() const
        {
            // TODO: Think about how you want to print the dimensions
            std::cout << "Tensor(shape={";
            for(size_t i=0; i < shape_.size(); ++i) 
                std::cout << shape_[i] << (i == shape_.size()-1 ? "" : ", ");
            std::cout << "})  ";
            
            // Note: Simple flat print. For deep learning, you'll want 
            // a more sophisticated nested loop print for matrices.
            for (size_t i = 0; i < total_size_; ++i) {
                std::cout << data_[i] << " ";
            }
            std::cout << std::endl;
        }
    };

    inline size_t shape_length(const std::vector<size_t>& shape)
    {
        return std::accumulate(
            shape.begin(), shape.end(),
            size_t{1},
            std::multiplies<size_t>()
        );
    }
}