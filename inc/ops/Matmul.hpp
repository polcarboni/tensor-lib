#pragma once
#include "core/Tensor.hpp"

namespace tensor
{

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
    

    Tensor matmul(const Tensor& lhs, const Tensor& rhs);

} // namespace tensor 