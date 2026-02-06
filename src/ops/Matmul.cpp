#include "ops/Backend.hpp"
#include "ops/Matmul.hpp"

namespace tensor
{
    std::vector<size_t> infer_matmul_shape(const std::vector<size_t>& lhs_shape,
                                        const std::vector<size_t>& rhs_shape,
                                        bool trans_a,
                                        bool trans_b)
    {

    }

    Tensor MatMulDispatcher::call(const Tensor& lhs, const Tensor& rhs, const MatMulParams& params)
    {

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

} // namespace tensor