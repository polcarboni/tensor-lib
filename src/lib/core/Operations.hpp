/*
    OPERATIONS TO BE IMPLEMENTED:
    - Matmul   (BINARY)
    - Element-wise: add, subtract, multiply, divide  (BINARY)
    
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

namespace tensor 
{

    // ------------------------------------------------------------------------------------------------------
    //                                       ELEMENT-WISE OPERATIONS 
    // ------------------------------------------------------------------------------------------------------ 
    

    template<typename T, typename Op>
    Tensor<T> element_wise(const Tensor<T>& a, const Tensor<T>& b, Op op)
    {
        if (a.shape() != b.shape())
        {
            //TODO: more informative error message (requires shape_to_string method)
            throw std::invalid_argument("Tensor shape mismatch");
        }

        //TODO-fix: start from empty tensor of the correct shape
        Tensor<T> result = a;

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


    // ------------------ INPLACE ELEMENT-WISE OPERATIONS ------------------
    // +=, -=, *=, /=
    // ... also require a generic element_wise_inplace base function ...


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

    }

    /*  COPIED FROM AI - just as a note
    // Normal version: returns a new tensor
    template <typename T>
    Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {
        // Check shapes: a.cols must equal b.rows
        Tensor<T> result(a.rows(), b.cols());
        matmul_to(a, b, result); // Call the worker function
        return result;
    }

    // Performance version: uses a pre-allocated result tensor
    template <typename T>
    void matmul_to(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& result) {
        if (a.cols() != b.rows()) throw std::invalid_argument("Incompatible shapes");
        
        // High-performance nested loops or BLAS call here
        for (size_t i = 0; i < a.rows(); ++i) {
            for (size_t j = 0; j < b.cols(); ++j) {
                T sum = 0;
                for (size_t k = 0; k < a.cols(); ++k) {
                    sum += a(i, k) * b(k, j);
                }
                result(i, j) = sum;
            }
        }
    }
    */


    // ------------------------------------------------------------------------------------------------------
    //                                        REDUCTION OPERATIONS
    // ------------------------------------------------------------------------------------------------------ 
    // - Reductions: sum, mean, max, min    (UNARY)


    // Global reduction vs axis reduction (global if axis not provided))
    
    /*  TOTAL REDUCTION OPERATIONS - AI generated
                
                template<typename T>
            T sum(const Tensor<T>& t) {
                if (t.size() == 0) return T{0};
                return std::accumulate(t.data(), t.data() + t.size(), T{0});
            }

            template<typename T>
            T max(const Tensor<T>& t) {
                if (t.size() == 0) throw std::runtime_error("Reduction on empty tensor");
                return *std::max_element(t.data(), t.data() + t.size());
            }

            template<typename T>
            T min(const Tensor<T>& t) {
                if (t.size() == 0) throw std::runtime_error("Reduction on empty tensor");
                return *std::min_element(t.data(), t.data() + t.size());
            }

            template<typename T>
            double mean(const Tensor<T>& t) {
                if (t.size() == 0) return 0.0;
                return static_cast<double>(sum(t)) / t.size();
            }
    */


    /*  AXIS_WISE REDUCTIONS  - AI generated

                template<typename T>
            Tensor<T> sum(const Tensor<T>& t, size_t axis) {
                const auto& old_shape = t.shape();
                if (axis >= old_shape.size()) throw std::out_of_range("Axis out of bounds");

                // 1. Compute new shape (remove the dimension at 'axis')
                std::vector<size_t> new_shape;
                for (size_t i = 0; i < old_shape.size(); ++i) {
                    if (i != axis) new_shape.push_back(old_shape[i]);
                }
                
                // Handle case where we reduce a 1D tensor to a scalar (0D tensor)
                if (new_shape.empty()) return Tensor<T>::zeros({1}); 

                Tensor<T> result(new_shape, T{0});
                const auto& old_strides = t.strides();
                const auto& new_strides = result.strides();

                // 2. Iterate through all elements of the original tensor
                for (size_t i = 0; i < t.size(); ++i) {
                    // Convert flat index 'i' to multi-index of output tensor
                    size_t remaining_flat_idx = 0;
                    size_t temp_idx = i;
                    
                    size_t out_dim_count = 0;
                    for (size_t d = 0; d < old_shape.size(); ++d) {
                        size_t coord = (temp_idx / old_strides[d]);
                        temp_idx %= old_strides[d];
                        
                        if (d != axis) {
                            remaining_flat_idx += coord * new_strides[out_dim_count++];
                        }
                    }
                    result[remaining_flat_idx] += t[i];
                }

                return result;
            }

            template<typename T>
            Tensor<double> mean(const Tensor<T>& t, size_t axis) {
                auto s = sum(t, axis);
                size_t divisor = t.shape()[axis];
                
                // Create a double tensor for result
                std::vector<size_t> res_shape = s.shape();
                Tensor<double> res(res_shape);
                for(size_t i = 0; i < s.size(); ++i) {
                    res[i] = static_cast<double>(s[i]) / divisor;
                }
                return res;
            }
    
    */
};