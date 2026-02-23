#include "core/TensorImpl.hpp"
#include "core/TensorIterator.hpp"
#include <utility>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace tensor::ops
{

    /**
     * TODO:
     * these require explicit instantiation for all the different operations.
     * Consider moving them back to .hpp or find another solution:
     * 
     * Explicit instantiation of all of them is not achievable. I have to provide that sort of
     * table for explicit instantiation since this is going to be also required by the TensorIterator.
     */

    // -------------------------------------------------------------------------------------------------------------  
    //                                               DISPATCHER IMPLEMENTATION
    // -------------------------------------------------------------------------------------------------------------  

    template <typename Op, typename... Args>
    void dispatch_impl_(TensorIterator& iter, Args&&... args)
    {
        DISPATCH_ALL_TYPES(iter.get_common_dtype(), "dispatch", [&]
        {
            if (iter.get_common_device().type == DeviceType::CPU) {
                Op::template cpu<scalar_t>(iter, std::forward<Args>(args)...);
            }
            #ifdef USE_CUDA
            else if (iter.get_common_device().type == DeviceType::CUDA) {
                
                // Current support only of single stream
                cudaStream_t stream = cudaStream_t(0);
                Op::template cuda<scalar_t>(iter, stream, std::forward<Args>(args)...);
            }
            #endif
            else {
                throw std::runtime_error("dispatch: Unsupported device " + to_string(iter.get_common_device()));
            }
        });
    }

    // template <typename Op, typename... Args>
    // TensorImpl dispatch_nullary(Device device, ScalarType dtype, std::vector<size_t>& shape, Args... args)
    // {
    //     TensorIterator iter;
    //     iter.build<Op>(shape);
    //     dispatch_impl_<Op>(iter, args...);

    //     return iter.get_output();
    // }


    /**
     * TODO: check which templates require explicit instantiations
     * 
     * TODO: check the correct use of return functions. return_output could be instead return_outputs[0] since
     *       TensorIterators generally suppots multiple output functions. A conveniency function for forward
     *       operations might not be worth the use.
     */
    
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
    

    // THE ONLY CORRECT ONE (FIX THE OTHER ONES)
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
    TensorImpl dispatch_reduction(TensorImpl& tensor, Args&&... args)
    {
        return TensorImpl{}; // placeholder
    }

} // namespace tensor::ops