#pragma once
#include "Types.hpp"
#include "TensorIterator.hpp"

namespace tensor
{
    struct TensorImpl;
}

// Check how inplace operations affect the dispatchers 
// Check the dispatchers interfaces (easy to do when using them in TensorImpl calls)

namespace tensor::ops
{

    template <typename Op, typename... Args>
    void dispatch_impl_(tensor::TensorIterator& iter, Args&&... args)
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

    template <typename Op, typename... Args>
    tensor::TensorImpl dispatch_nullary(Device device, ScalarType dtype, std::vector<size_t>& shape, Args... args)
    {
        tensor::TensorIterator iter;
        iter.build<Op>(shape);
        dispatch_impl_<Op>(iter, args...);

        return iter.get_output();
    }

    
    template <typename Op, typename... Args>
    tensor::TensorImpl dispatch_unary(Device device, ScalarType dtype, TensorImpl& in, Args... args);
    // {
    //     // tensor::TensorIterator iter;
    //     // iter.add_input(in);
    //     // iter.build<Op>();

    //     // dispatch_impl_<Op>(device, dtype, iter, args...);
    //     // return iter.get_output();
    // }
    

    template <typename Op, typename... Args>
    void dispatch_unary_inplace(TensorImpl& in, Args&&... args)
    {   
        tensor::TensorIterator iter;
        iter.add_output(&in);
        iter.add_input(&in);
        iter.build<Op>();

        dispatch_impl_<Op>(iter, std::forward<Args>(args)...);
    }

    template <typename Op, typename... Args>
    tensor::TensorImpl dispatch_unary_casting(Device device, ScalarType dtype, TensorImpl& in, Args... args);
    // {   
    //     // tensor::TensorIterator iter;
    //     // iter.add_input(in);
    //     // iter.build<Op>(dtype);

    //     // dispatch_impl_<Op>(iter, args...);
    //     // return iter.get_output();
    // }
    

    template <typename Op, typename... Args>
    tensor::TensorImpl dispatch_binary(Device device, ScalarType dtype,
        tensor::TensorImpl& lhs, tensor::TensorImpl& rhs, Args... args)
    {
        tensor::TensorIterator iter;
        iter.add_input(lhs);
        iter.add_input(rhs);
        iter.build<Op>();

        dispatch_impl_<Op>(iter, args...);
        return iter.get_output();
    }

    template <typename BackwardOp, typename... Args>
    void dispatch_binary_backward(Device device, ScalarType dtype,
        tensor::TensorImpl& output, tensor::TensorImpl& lhs, tensor::TensorImpl& rhs,
        Args... args)
    {
        tensor::TensorIterator iter;
        iter.add_input(output.autograd_meta_.grad_);    //Upstream grad
        iter.add_input(lhs);
        iter.add_input(rhs);
        iter.add_output(lhs.autograd_meta_.grad_);
        iter.add_output(rhs.autograd_meta_.grad_);
        iter.build<BackwardOp>(device, dtype);

        dispatch_impl_<BackwardOp>(device, dtype, iter, args...);
        return iter.get_outputs();  // Return both output tensors?
    }
    
    template <typename Op, typename... Args>
    void dispatch_ternary(Device device, ScalarType dtype, TensorImpl* a, TensorImpl* b, TensorImpl* c, Args... args);
    
    template <typename Op, typename... Args>
    void dispatch_comparison(Device device, ScalarType dtype, TensorImpl* lhs, TensorImpl* rhs, Args... args);
    
    template <typename Op, typename... Args>
    void dispatch_reduction(Device device, ScalarType dtype, TensorImpl* tensor, Args... args);

} // namespace tensor::ops