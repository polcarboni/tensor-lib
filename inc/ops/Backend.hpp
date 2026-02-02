#pragma once
#include "core/Tensor.hpp"
#include "TensorIterator.hpp"

#define USE_CUDA

namespace tensor
{
    // -------------------------------------------------------------------------------------------------------------  
    //                                           BACKEND KERNEL INTERFACES
    // ------------------------------------------------------------------------------------------------------------- 

    // ----------------------------------------- linear algebra (matmul) dispatcher --------------------------------
    
    struct MatMulParams {
        bool trans_a = false;
        bool trans_b = false;
        double alpha = 1.0;
        double beta = 0.0;
    };
    
    struct MatMulDispatcher {
        static Tensor call(const Tensor& lhs, const Tensor& rhs, const MatMulParams& params = {});
    };

    std::vector<size_t> infer_matmul_shape(const std::vector<size_t>& lhs_shape,
                                           const std::vector<size_t>& rhs_shape,
                                           bool trans_a = false,
                                           bool trans_b = false);
    
    
    struct CPUDevice {
        template<typename scalar_t, typename Op>
        static void launch_nullary(TensorIterator& iter, const Op& op);

        template<typename scalar_t, typename Op>
        static void launch_unary(TensorIterator& iter, const Op& op);
    
        template<typename scalar_t, typename Op>
        static void launch_binary(TensorIterator& iter, const Op& op);
        
        template<typename scalar_t, typename Op>
        static void launch_ternary(TensorIterator& iter, const Op& op);
        
        /* Requires two types scalar type and accumulator type (wider)*/
        template<typename scalar_t, typename acc_t, typename Op>
        static void launch_reduction(TensorIterator& iter, const Op& op, acc_t identity);

        template<typename scalar_t>
        static void launch_matmul(Tensor& out, const Tensor& lhs, const Tensor& rhs, const MatMulParams& params);
    };    

    #ifdef USE_CUDA
        struct CUDADevice {
            template<typename scalar_t, typename Op>
            static void launch_nullary(TensorIterator& iter, const Op& op);

            template<typename scalar_t, typename Op>
            static void launch_unary(TensorIterator& iter, const Op& op);
        
            template<typename scalar_t, typename Op>
            static void launch_binary(TensorIterator& iter, const Op& op);
            
            template<typename scalar_t, typename Op>
            static void launch_ternary(TensorIterator& iter, const Op& op);
            
            /* Requires two types scalar type and accumulator type (wider)*/
            template<typename scalar_t, typename acc_t, typename Op>
            static void launch_reduction(TensorIterator& iter, const Op& op, acc_t identity);

            template<typename scalar_t>
            static void launch_matmul(Tensor& out, const Tensor& lhs, const Tensor& rhs, const MatMulParams& params);
        };
    #endif

} // namespace tensor