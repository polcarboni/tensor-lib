#pragma once
#include "core/Tensor.hpp"
#include "Dispatcher.hpp"
#include "TensorIterator.hpp"
#include "Backend.hpp"

namespace tensor
{
    std::vector<size_t> broadcast_shapes(const std::vector<size_t>&, const std::vector<size_t>&);

    template <template <typename> class Op>
    Tensor nullary_op_impl(std::vector<size_t> shape, ScalarType dtype, Device device, const std::string& op_name)
    {
        Tensor result(shape, dtype, device);

        TensorIteratorConfig config;
        config.add_output(result);
        auto iter = TensorIterator::build(config);

        NullaryElementwiseDispatcher<Op>::call(iter, op_name);

        return result;
    }


    template <template <typename> class Op>
    Tensor unary_op_impl(const Tensor& lhs, const std::string& op_name)
    {
        Tensor result(lhs.shape(), lhs.dtype(), lhs.device());

        TensorIteratorConfig config;
        config.add_output(result).add_input(lhs);
        auto iter = TensorIterator::build(config);

        UnaryElementwiseDispatcher<Op>::call(iter, op_name);

        return result;
    }


    template <template <typename> class Op>
    Tensor binary_op_impl(const Tensor& lhs, const Tensor& rhs, const std::string& op_name)
    {
        if (lhs.device() != rhs.device()) {
            throw std::runtime_error(op_name + ": device mismatch");
        }
        auto out_shape = broadcast_shapes(lhs.shape(), rhs.shape());
        ScalarType out_dtype = promote_types(lhs.dtype(), rhs.dtype());
        Tensor result(out_shape, out_dtype, lhs.device());

        TensorIteratorConfig config;
        config.add_output(result).add_input(lhs).add_input(rhs);
        auto iter = TensorIterator::build(config);    
        
        BinaryElementwiseDispatcher<Op>::call(iter, op_name);
        
        return result;
    }

    template <template <typename> class Op, typename acc_t>
    Tensor reduction_op_impl(const Tensor& lhs, std::vector<size_t> dims, bool keepdim,
                             acc_t identity, const std::string& name)
    {
        auto dummy = Tensor();
        return dummy;
    }

} // namespace tensor