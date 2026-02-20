#pragma once
#include "Types.hpp"
#include "TensorIterator.hpp"


namespace tensor::ops
{

    // -------------------------------------------------------------------------------------------------------------  
    //                                               DISPATCHER IMPLEMENTATION
    // -------------------------------------------------------------------------------------------------------------  

    template <typename Op, typename... Args>
    void dispatch_impl_(TensorIterator& iter, Args&&... args)
    {
        DISPATCH_ALL_TYPES(iter.get_common_dtype(), "dispatch", [&]
        {
            if (iter.get_common_device() == DeviceType::CPU) {
                Op::template cpu<scalar_t>(iter, std::forward<Args>(args)...);
            }
            #ifdef USE_CUDA
            else if (iter.get_common_device() == DeviceType::CUDA) {
                auto stream = get_current_cuda_stream();
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
     * 
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
        iter.add_input(in);
        iter.build<Op>();

        dispatch_impl_<Op>(iter, std::forward<Args>(args)...);
        return iter.get_output();
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
        iter.add_input(in);
        iter.build<Op>(dtype);

        dispatch_impl_<Op>(iter, std::forward<Args>(args)...);
        return iter.get_output();
    }
    

    // -------------------------------------------------------------------------------------------------------------  
    //                                                 BINARY DISPATCHERS
    // -------------------------------------------------------------------------------------------------------------      

    template <typename Op, typename... Args>
    TensorImpl dispatch_binary(TensorImpl& lhs, TensorImpl& rhs, Args&&... args)
    {
        TensorIterator iter;
        iter.add_input(lhs);
        iter.add_input(rhs);
        iter.build<Op>();

        dispatch_impl_<Op>(iter, std::forward<Args>(args)...);
        return iter.get_output();
    }

    // TODO-fix: this return type should probably be a pair or a vector. Surely not void.
    template <typename BackwardOp, typename... Args>
    void dispatch_binary_backward(Device device, ScalarType dtype,
        TensorImpl& output, TensorImpl& lhs, TensorImpl& rhs,
        Args&&... args)
    {
        TensorIterator iter;
        iter.add_input(output.autograd_meta_.grad_);    //Upstream grad
        iter.add_input(lhs);
        iter.add_input(rhs);
        iter.add_output(lhs.autograd_meta_.grad_);
        iter.add_output(rhs.autograd_meta_.grad_);
        iter.build<BackwardOp>(device, dtype);

        dispatch_impl_<BackwardOp>(device, dtype, iter, std::forward<Args>(args)...);
        return iter.get_outputs();  // Return both output tensors?
    }


    // -------------------------------------------------------------------------------------------------------------  
    //                                                 TERNARY DISPATCHERS
    // -------------------------------------------------------------------------------------------------------------  

    template <typename Op, typename... Args>
    void dispatch_ternary(Device device, ScalarType dtype, TensorImpl* a, TensorImpl* b, TensorImpl* c, Args&&... args)
    {

    }
    

    // -------------------------------------------------------------------------------------------------------------  
    //                                                COMPARISON DISPATCHERS
    // ------------------------------------------------------------------------------------------------------------- 

    template <typename Op, typename... Args>
    void dispatch_comparison(Device device, ScalarType dtype, TensorImpl* lhs, TensorImpl* rhs, Args&&... args)
    {

    }

    
    // -------------------------------------------------------------------------------------------------------------  
    //                                                 REDUCTION DISPATCHERS
    // ------------------------------------------------------------------------------------------------------------- 

    template <typename Op, typename... Args>
    void dispatch_reduction(Device device, ScalarType dtype, TensorImpl* tensor, Args... args)
    {

    }

} // namespace tensor::ops