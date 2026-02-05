// #include "core/Tensor.hpp"
// #include "ops/Functors.hpp"
// #include "ops/OpsImpl.hpp"

#include "TensorOps.hpp"

namespace tensor
{
    // ----------------------- operations api ----------------------- 
    
    // Binary operations
    Tensor add(const Tensor& lhs, const Tensor& rhs)
    {
        return binary_op_impl<AddFunctor>(lhs, rhs, "add");
    }

    Tensor mul(const Tensor& lhs, const Tensor& rhs)
    {
        return binary_op_impl<MulFunctor>(lhs, rhs, "mul");
    }


    // unary ReLU
    Tensor ReLU(const Tensor& lhs)
    {
        return unary_op_impl<ReLUFunctor>(lhs, "relu");
    }
    
    // reduction op
    Tensor sum(const Tensor& lhs, std::vector<size_t> dims, bool keepdim) {
        return reduction_op_impl<AddFunctor>(lhs, dims, keepdim, 0.0, "sum");
    }

    Tensor mean(const Tensor& lhs, std::vector<size_t> dims, bool keepdim)
    {
        Tensor result = sum(lhs, dims, keepdim);

        // // TODO: provide these implementation
        // double count = calculate_reduction_count(lhs, dims);
        // return result / count;

        return result;
    }

    // // ----------------------- operators oveloading ----------------------- 
    
    // inline Tensor operator+(const Tensor& lhs, const Tensor& rhs)
    // {
    //     return add(lhs,rhs);
    // }

    // inline Tensor operator*(const Tensor& lhs, const Tensor& rhs)
    // {
    //     return mul(lhs, rhs);
    // }
 
} // namespace tensor