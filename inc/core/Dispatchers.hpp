#include "core/TensorImpl.hpp"
#include "core/TensorIterator.hpp"
#include <utility>
#include <optional>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace tensor::ops
{


    // -------------------------------------------------------------------------------------------------------------  
    //                                               DISPATCHER IMPLEMENTATION
    // -------------------------------------------------------------------------------------------------------------  

    template <typename Op, typename... Args>
    void dispatch_impl_(TensorIterator& iter, Args&&... args)
    {
            if (iter.get_common_device().type == DeviceType::CPU) {
                Op::cpu(iter, std::forward<Args>(args)...);
            }
            
            #ifdef USE_CUDA
                else if (iter.get_common_device().type == DeviceType::CUDA) {
                    cudaStream_t stream = cudaStream_t(0);                      // Current support only of single stream
                    Op::cuda(iter, stream, std::forward<Args>(args)...);
                }
            #endif

            else {
                throw std::runtime_error("dispatch: Unsupported device " + to_string(iter.get_common_device()));
            }
    }

    
    // -------------------------------------------------------------------------------------------------------------  
    //                                                  UNARY DISPATCHERS
    // -------------------------------------------------------------------------------------------------------------  
    
    template <typename Op, typename... Args>
    TensorImpl dispatch_unary(TensorImpl& in, Args&&... args)
    {
        TensorIterator iter;
        iter.add_input(&in);
        iter.build<Op>();

        dispatch_impl_<Op>(iter, std::forward<Args>(args)...);
        return *iter.get_outputs()[0];
    }
    
    template <typename Op, typename... Args>
    void dispatch_unary_inplace(TensorImpl& in, Args&&... args)
    {   
        TensorIterator iter;
        iter.add_output(&in);
        iter.add_input(&in);
        iter.set_inplace(true);
        iter.build<Op>();

        dispatch_impl_<Op>(iter, std::forward<Args>(args)...);
    }

    template <typename Op, typename... Args>
    TensorImpl dispatch_unary_casting(TensorImpl& in, ScalarType dtype, Args&&... args)
    {   
        TensorIterator iter;
        iter.add_input(&in);
        iter.build<Op>(dtype);

        dispatch_impl_<Op>(iter, std::forward<Args>(args)...);
        return *iter.get_outputs()[0];
    }
    

    // -------------------------------------------------------------------------------------------------------------  
    //                                                 BINARY DISPATCHERS
    // -------------------------------------------------------------------------------------------------------------      

    template <typename Op, typename... Args>
    TensorImpl dispatch_binary(TensorImpl& lhs, TensorImpl& rhs, Args&&... args)
    {
        TensorIterator iter;
        iter.add_input(&lhs);
        iter.add_input(&rhs);
        iter.build<Op>();

        dispatch_impl_<Op>(iter, std::forward<Args>(args)...);
        return *iter.get_outputs()[0];
    }

    template <typename BackwardOp, typename... Args>
    ::std::pair<TensorImpl, TensorImpl> dispatch_binary_backward(TensorImpl& lhs, TensorImpl& rhs, Args&&... args)
    {
        TensorIterator iter;
        // iter.add_input(output.autograd_meta_.grad_);    //Upstream grad
        iter.add_input(&lhs);
        iter.add_input(&rhs);
        // iter.add_output(lhs.autograd_meta_.grad_);
        // iter.add_output(rhs.autograd_meta_.grad_);
        iter.build<BackwardOp>();

        dispatch_impl_<BackwardOp>(iter, std::forward<Args>(args)...);
        return {TensorImpl{}, TensorImpl{}};    //placeholder
    }


    // -------------------------------------------------------------------------------------------------------------  
    //                                                 TERNARY DISPATCHERS
    // -------------------------------------------------------------------------------------------------------------  

    template <typename Op, typename... Args>
    TensorImpl dispatch_ternary(TensorImpl& op_a, TensorImpl& op_b, TensorImpl& op_c, Args&&... args)
    {
        return TensorImpl{}; // placeholder
    }
    

    // -------------------------------------------------------------------------------------------------------------  
    //                                                COMPARISON DISPATCHERS
    // ------------------------------------------------------------------------------------------------------------- 

    template <typename Op, typename... Args>
    TensorImpl dispatch_comparison(TensorImpl& lhs, TensorImpl& rhs, Args&&... args)
    {
        return TensorImpl{}; // placeholder
    }

    
    // -------------------------------------------------------------------------------------------------------------  
    //                                                 REDUCTION DISPATCHERS
    // ------------------------------------------------------------------------------------------------------------- 

    template <typename Op, typename... Args>
    TensorImpl dispatch_reduction(TensorImpl& tensor,
        const std::optional<std::vector<size_t>> axes,
        bool keepdims, Args&&... args)
    {
        TensorIterator iter;
        iter.add_input(&tensor);
        iter.set_reduction_axes(atd::move(axes));
        iter.set_keepdims(keepdims);
        iter.build<Op>();

        dispatch_impl_<Op>(iter, std::forward<Args>(args)...);
        
        return *iter.get_outputs()[0];
    }

    // Inplace reduction to a provided output tensor
    template <typename Op, typename... Args>
    void dispatch_reduction_inplace(TensorImpl& out, TensorImpl& in,
        const std::optional<std::vector<size_t>> axes,
        bool keepdims, Args&&... args)
    {
        TensorIterator iter;
        iter.add_input(&in);
        iter.add_output(&out);
        iter.set_reduction_axes(std::move(axes));
        iter.set_keepdims(keepdims);
        iter.set_inplace(true);
        
        iter.build<Op>();

        dispatch_impl_<Op>(iter, std::forward<Args>(args)...);
    }



} // namespace tensor::ops