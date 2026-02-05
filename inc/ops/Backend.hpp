#pragma once
#include "core/Tensor.hpp"
#include "TensorIterator.hpp"
#include "ops/Matmul.hpp"

namespace tensor
{
    // -------------------------------------------------------------------------------------------------------------  
    //                                           BACKEND KERNEL INTERFACES
    // ------------------------------------------------------------------------------------------------------------- 
    
    struct CPUDevice {
        template<typename scalar_t, typename Op>
        static void launch_nullary(TensorIterator& iter, const Op& op)
        {

        }

        template<typename scalar_t, typename Op>
        static void launch_unary(TensorIterator& iter, const Op& op)
        {

        }
    
        template<typename scalar_t, typename Op>
        static void launch_binary(TensorIterator& iter, const Op& op)
        {

        }
        
        template<typename scalar_t, typename Op>
        static void launch_ternary(TensorIterator& iter, const Op& op)
        {

        }
        
        /* Requires two types scalar type and accumulator type (wider)*/
        template<typename scalar_t, typename acc_t, typename Op>
        static void launch_reduction(TensorIterator& iter, const Op& op, acc_t identity)
        {

        }

        template<typename scalar_t>
        static void launch_matmul(Tensor& out, const Tensor& lhs, const Tensor& rhs, const MatMulParams& params)
        {

        }
    };    

    #ifdef USE_CUDA
        struct CUDADevice {
            template<typename scalar_t, typename Op>
            static void launch_nullary(TensorIterator& iter, const Op& op)
            {

            }

            template<typename scalar_t, typename Op>
            static void launch_unary(TensorIterator& iter, const Op& op)
            {

            }
        
            template<typename scalar_t, typename Op>
            static void launch_binary(TensorIterator& iter, const Op& op)
            {

            }
            
            template<typename scalar_t, typename Op>
            static void launch_ternary(TensorIterator& iter, const Op& op)
            {

            }
            
            /* Requires two types scalar type and accumulator type (wider)*/
            template<typename scalar_t, typename acc_t, typename Op>
            static void launch_reduction(TensorIterator& iter, const Op& op, acc_t identity)
            {
                
            }

            template<typename scalar_t>
            static void launch_matmul(Tensor& out, const Tensor& lhs, const Tensor& rhs, const MatMulParams& params)
            {

            }
        };
    #endif

} // namespace tensor