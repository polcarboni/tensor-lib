#include "core/TensorImpl.hpp"
#include "core/TensorIterator.hpp"
#include "ops/BinaryOps.hpp"
#include <cstdint>
#include <cassert>
#include <cmath>

namespace tensor::ops::kernel {

    template <typename Op>
    void binary_cpu_kernel(TensorIterator& iter, Op op)
    {
        auto dtype = iter.get_common_dtype();

        DISPATCH_ALL_TYPES(dtype, "binary_cpu_kernel", ([&] {
            scalar_t* output    = iter.output_ptr<scalar_t>(0);
            const scalar_t* lhs = iter.input_ptr<scalar_t>(0);
            const scalar_t* rhs = iter.input_ptr<scalar_t>(1);

            const auto& lhs_strides = iter.get_strides(0);
            const auto& rhs_strides = iter.get_strides(1);
            const auto& out_strides = iter.get_strides(2);
            
            const auto& is_broadcasted = iter.get_is_broadcasted();
            size_t numel      = iter.get_numel();
            const auto& shape = iter.get_shape();
            size_t ndim       = iter.get_ndim();

            if (iter.get_common_is_contiguous() && ndim == 1 && !is_broadcasted) {
                /* Contiguous elements and single dimension coalesced */
                for (size_t i = 0; i < numel; ++i) {
                    output[i] = op(lhs[i], rhs[i]);
                }
            } else if (ndim == 1) {
                /* non contiguous, single dimension */
                for (size_t i = 0; i < numel; ++i) {
                    output[i * out_strides[0]] = op(lhs[i * lhs_strides[0]], rhs[i * rhs_strides[0]]);
                }
            } else {
                /* General path */
                std::vector<size_t> counter(ndim, 0);
                size_t lhs_offset = 0;
                size_t rhs_offset = 0;
                size_t out_offset = 0;
                
                for (size_t i = 0; i < numel; ++i) {
                    
                    output[out_offset] = op(lhs[lhs_offset], rhs[rhs_offset]);
                    
                    for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                        ++counter[d];
                        lhs_offset += lhs_strides[d];
                        rhs_offset += rhs_strides[d];
                        out_offset += out_strides[d];

                        if (counter[d] < shape[d]) break;

                        /* Reset offset and counter when the dimension is computed */
                        counter[d] = 0;
                        lhs_offset -= shape[d] * lhs_strides[d];
                        rhs_offset -= shape[d] * rhs_strides[d];
                        out_offset -= shape[d] * out_strides[d];
                    }
                }
            }    
        }));
    };

}; // namespace tensor::ops::kernel


namespace tensor::ops
{

    void BinaryAdd::cpu(TensorIterator& iter) {
        kernel::binary_cpu_kernel(iter, [](auto a, auto b) { return a + b; });
    }
    
    void BinaryAddBackward::cpu(TensorIterator& iter) {
        // PLACEHOLDER
    }
    
    void BinarySub::cpu(TensorIterator& iter) {
        kernel::binary_cpu_kernel(iter, [](auto a, auto b) { return a - b; });
    }
    
    void BinaryExp::cpu(TensorIterator& iter) {
        kernel::binary_cpu_kernel(iter, [](auto a, auto b) { return std::pow(a,b); });
    }
    
} // namespace tensor::ops