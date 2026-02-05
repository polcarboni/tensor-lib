#pragma once
#include "ops/Backend.hpp"

namespace tensor
{
    class CUDADevice {
    public:
        template <typename scalar_t, typename Op>
        static void launch_nullary(TensorIterator& iter, const Op& op);
        
        template <typename scalar_t, typename Op>
        static void launch_unary(TensorIterator& iter, const Op& op);
    };
} // namespace tensor