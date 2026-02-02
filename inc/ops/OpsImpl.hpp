#pragma once
#include "core/Tensor.hpp"
#include "Dispatcher.hpp"
#include "TensorIterator.hpp"
#include "Backend.hpp"

namespace tensor
{
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


    // <!> for size > 2 the matmul operation is batched
    Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
        if( lhs.dims() < 2 || rhs.dims() < 2) {
            throw std::runtime_error("matmul: tensors must be at least 2D");
        }

        int64_t M = lhs.shape()[lhs.dims() - 2];
        int64_t K1 = lhs.shape()[lhs.dims() - 1];
        int64_t K2 = rhs.shape()[lhs.dims() - 2];
        int64_t N = rhs.shape()[lhs.dims() - 1];

        if (K1 != K2) {
            throw std::runtime_error("matmul: size_mismatch");
        }

        auto out_shape = infer_matmul_shape(lhs.shape(), rhs.shape());

        //TODO-fix bug: the result is defined but never used
        Tensor result(out_shape, promote_types(lhs.dtype(), rhs.dtype()), lhs.device());

        // TODO-fix: check the correct use of this
        return MatMulDispatcher::call(lhs, rhs);;
    }


    template <template <typename> class Op, typename acc_t>
    Tensor reduction_op_impl(const Tensor& lhs, std::vector<size_t> dims, bool keepdim,
                             acc_t identity, const std::string& name);

} // namespace tensor