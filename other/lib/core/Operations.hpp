/*
    OPERATIONS TO BE IMPLEMENTED:
    - Matmul   (BINARY)
    - Element-wise: add, subtract, multiply, divide  (BINARY)
    - Reductions: sum, mean, max, min    (UNARY)  global/to axis
    - Dot product (BINARY) ?: flatten matrix and compute dot-product (Y, forbenius innermost product: reduction + vector dot product)
    - Broadcasting  (N-ARY helper)
    - Batch Matrix multiplication FOR DL batch processing (BINARY)
    - Linear system solver (BINARY)
    - Matrix decomposition: LU, QR, SVD, Cholesky (UNARY)
    - Eigenvalue decomposition: Spectral, analysis, PCA (UNARY)
    - Determinant, inverse, trace (UNARY)
    

    (The list might not be 100% correct)
*/

#pragma once
#include "Tensor.hpp"
#include <cmath>
#include <algorithm>
// #include <execution>

namespace tensor_old 
{

    // ------------------------------------------------------------------------------------------------------
    //                                       ELEMENT-WISE OPERATIONS 
    // ------------------------------------------------------------------------------------------------------ 
    

    template<typename T, typename Op>
    Tensor<T> element_wise(const Tensor<T>& a, const Tensor<T>& b, Op op)
    {
        if (a.shape() != b.shape())
        {
            throw std::invalid_argument("Tensor shape mismatch:" +
                                        shape_to_string(a.shape()) +
                                        " - " + shape_to_string(b.shape()));
        }

        Tensor<T> result(a.shape());

        //TODO: add call for parallel execution
        for (size_t i = 0; i < a.size(); ++i)
        {
            result[i] = op(a[i], b[i]);
        }

        return result;
    } 

    template <typename T>
    Tensor<T> operator+(const Tensor<T>& a, const Tensor<T>& b)
    {
        return element_wise(a, b, std::plus<T>{});
    }
    
    template <typename T>
    Tensor<T> operator-(const Tensor<T>& a, const Tensor<T>& b)
    {
        return element_wise(a, b, std::minus<T>{});
    }
    
    template <typename T>
    Tensor<T> operator*(const Tensor<T>& a, const Tensor<T>& b)
    {
        return element_wise(a, b, std::multiplies<T>{});
    }

    template <typename T>
    Tensor<T> operator/(const Tensor<T>& a, const Tensor<T>& b)
    {
        return element_wise(a, b, std::divides<T>{});
    }


    // --------------------------------- INPLACE ELEMENT-WISE OPERATIONS  -----------------------------------
    // +=, -=, *=, /=
    // ... also require a generic element_wise_inplace base function ...

    template <typename T, typename Op>
    void element_wise_inplace(Tensor<T>& a, const Tensor<T>&b, Op op)
    {
        if(a.shape() != b.shape()) {
            throw std::invalid_argument(
                "Tensor shape mismatch for += operation: " +
                shape_to_string(a.shape()) + " - " + shape_to_string(b.shape())
            );
        }

        for (size_t i = 0; i < a.size(); ++i) {
            a[i] = op(a[i], b[i]);
        }
    } 

    template <typename T>
    Tensor<T>& Tensor<T>::operator+=(const Tensor<T>& other) {
        element_wise_inplace(*this, other, std::plus<T>());
    }

    // ------------------------------------------------------------------------------------------------------
    //                                        MATRIX MULTIPLICATION
    // ------------------------------------------------------------------------------------------------------ 

    /* This section requires the definition of the operation for multiple shapes, not only for 2D,
       including broadcasting and other things I have to check more into detail.
       - Start with simple only 2D and 3D version.
       - Move to higher dimensions
       - Add dispatching (?) for different sizes (batch multiplication)
       - A single 'dispatcher' function can then choose the correct one for the provided input
    */

    template <typename T>
    Tensor<T> matmul_2D(const Tensor<T>& a, const Tensor<T>& b)
    {
        if (a.shape().size() != 2 || b.shape().size() != 2)
        {
            throw std::runtime_error("Matmul 2D only supprts 2D tensors");
        }

        auto rows_a = a.shape()[0];
        auto cols_a = a.shape()[1];
        auto rows_b = b.shape()[0];
        auto cols_b = b.shape()[1];

        // cols_a != rows_b
        if (cols_a != rows_b)
        {
            std::cerr << "ROWS A: " << rows_a << " - COLS B: " << cols_b;
            throw std::runtime_error("Incompatible shapes");
        }

        auto out = zeros<T>(rows_a, cols_b);
        
        //TODO-fix: inefficient implementation (call here a helper function instead)
        for (size_t i = 0; i < rows_a; ++i) {
            for (size_t j = 0; j < cols_b; ++j) {
                T sum = 0;
                for (size_t k = 0; k < cols_a; ++k) {
                    sum += a(i,k) * b(k,j);
                }
                out(i,j) = sum;
            }
        }
        return out;
    }

    //TODO: implementation of N-size tensor multiplication
    //TODO: dispatcher implementation (simply call matmul on any size)


    // ------------------------------------------------------------------------------------------------------
    //                                       UNARY OPERATIONS 
    // ------------------------------------------------------------------------------------------------------ 

    //UNARY MAPPER
    //TODO-fix: I do not like the name
    template<typename T, typename Op>
    Tensor<T> map(const Tensor<T>&a, Op op)
    {
        Tensor<T> result(a.shape());

        //TODO: add call for parallel application
        for (size_t i = 0; i < a.size(); ++i)
        {
            result[i] = op(a[i]);
        }
        return result;
    }

    //TODO-remove(?): inplace but not member function (not good API)
    // template<typename T, typename Op>
    // Tensor<T> map_inplace(const Tensor<T>&a, Op op)
    // {
    //     for (size_t i = 0; i < a.size(); ++i)
    //     {
    //         a[i] = func(a[i]);
    //     }
    // }

    template <typename T>
    template <typename Op>
    Tensor<T>& Tensor<T>::map_inplace(Op op) {
        T* ptr = this->data();

        std::transform(//std::execution::par,
                       ptr, ptr + this->total_size_,
                       ptr,
                       op);
        
        // Increase version for backpropagation
        this->version_++;
        return *this;
    }

    // ------------------------------------------------------------------------------------------------------
    //                                       ACTIVATION FUNCTIONS
    // ------------------------------------------------------------------------------------------------------ 

    template <typename T>
    Tensor<T> ReLU(const Tensor<T>& a)
    {
        return map(a, [](T x) { return x > 0 ? x : T{0}; });
    }

    template <typename T>
    Tensor<T> sigmoid(const Tensor<T>& a)
    {
        return map(a, [](T x) { return std::tanh(x); });
    }

    //TODO: implement the other activation functions
    //TODO-fix: add call to efficient parallel execution


    // ------------------------------------ INPLACE ACTIVATION FUNCTIONS ------------------------------------

    //TODO-fix (in Tensor.hpp): add the function declarations in the file and implement here the logic
    //                          include this file at the end of Tensor.hpp

    template <typename T>
    Tensor<T>& Tensor<T>::ReLU() {
        return this->map_inplace([](T x) { return x > 0 ? x : T{0}; });
    }


    // template<typename T>
    // void ReLU_inplace(Tensor<T>& a)
    // {
    //     map_inplace(a, [](T x){ return x > 0 ? x : 0; })
    // }

} // namespace tensor