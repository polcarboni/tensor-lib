#include "core/TensorImpl.hpp"
#include "core/TensorIterator.hpp"
#include "ops/UnaryOps.hpp"
#include <cstdint>
#include <cassert>
#include <vector>
#include <cstring>
#include <cmath>

namespace tensor::ops::kernel {

    template <typename Op>
    void unary_cpu_kernel(TensorIterator& iter, Op op)
    {
        auto dtype = iter.get_common_dtype();

        DISPATCH_ALL_TYPES(dtype, "unary_cpu_kernel", [&] {
            scalar_t* output    = iter.output_ptr<scalar_t>(0);
            const scalar_t* src = iter.input_ptr<scalar_t>(0);  // single input

            const auto& src_strides = iter.get_strides(0);
            const auto& out_strides = iter.get_strides(1);      // output is index 1, not 2

            const auto& is_broadcasted = iter.get_is_broadcasted();
            size_t numel      = iter.get_numel();
            const auto& shape = iter.get_shape();
            size_t ndim       = iter.get_ndim();

            if (iter.get_common_is_contiguous() && ndim == 1 && !is_broadcasted) {
                /* Contiguous elements and single dimension coalesced */
                for (size_t i = 0; i < numel; ++i) {
                    output[i] = op(src[i]);
                }
            } else if (ndim == 1) {
                /* Non-contiguous, single dimension */
                for (size_t i = 0; i < numel; ++i) {
                    output[i * out_strides[0]] = op(src[i * src_strides[0]]);
                }
            } else {
                /* General path */
                std::vector<size_t> counter(ndim, 0);
                size_t src_offset = 0;
                size_t out_offset = 0;

                for (size_t i = 0; i < numel; ++i) {

                    output[out_offset] = op(src[src_offset]);

                    for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                        ++counter[d];
                        src_offset += src_strides[d];
                        out_offset += out_strides[d];

                        if (counter[d] < shape[d]) break;

                        /* Reset offset and counter when the dimension is computed */
                        counter[d] = 0;
                        src_offset -= shape[d] * src_strides[d];
                        out_offset -= shape[d] * out_strides[d];
                    }
                }
            }
        });
    }

} // namespace tensor::ops::kernel

namespace tensor::ops {
    
    void UnaryCastOp::cpu(TensorIterator& iter)
    {
        auto to_dtype   = iter.get_common_dtype();
        auto from_dtype = iter.get_input_dtype();

        auto numel = iter.get_numel();

        DISPATCH_ALL_TYPES(from_dtype, "castOp", [&] {
            using SrcType = scalar_t;
            scalar_t* input = iter.input_ptr<SrcType>(0);
            DISPATCH_ALL_TYPES(to_dtype, "castOp", [&] {
                using DstType = scalar_t;
                scalar_t* output = iter.output_ptr<DstType>(0);

                for (size_t i = 0; i < numel; ++i) {
                    output[i] = static_cast<DstType>(input[i]);
                }

            });
        });
    }

    void ContiguousOp::cpu(TensorIterator& iter)
    {
        ScalarType dtype = iter.get_common_dtype();
        
        DISPATCH_ALL_TYPES(dtype, "ContiguousOp", [&] {
            scalar_t*       output = iter.output_ptr<scalar_t>(0);
            const scalar_t* input  = iter.input_ptr<scalar_t>(0);
            size_t numel = iter.get_numel();

            if (iter.get_common_is_contiguous()) {
                // Already contiguous input
                std::memcpy(output, input, numel * sizeof(scalar_t));
            } else {
                const std::vector<size_t>& shape = iter.get_shape();
                const auto& input_strides  = iter.get_strides(0);
                const auto& output_strides = iter.get_strides(1);
                size_t ndim = iter.get_ndim();

                if (ndim == 1) {
                    for (size_t i = 0; i < numel; ++i) {
                        output[i] = input[i * input_strides[0]];
                    }
                } else {
                    // n-dimensional tensor
                    std::vector<size_t> counter(ndim, 0);
                    size_t input_offset = 0;

                    // Sequential writing to output
                    for (size_t i = 0; i < numel; ++i) {
                        output[i] = input[input_offset];
                    }

                    // Stride-based input offset computation with carry
                    for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                        ++counter[d];
                        input_offset += input_strides[d];

                        if (counter[d] < shape[d]) break;

                        counter[d] = 0;
                        input_offset -= shape[d] * input_strides[d];
                    }
                }
            }
        });
    }

    void UnaryNeg::cpu(TensorIterator& iter)
    {
        if (iter.get_common_dtype() == ScalarType::Bool) {
            kernel::unary_cpu_kernel(iter, [](auto x) { return !x; });
        } else {
            kernel::unary_cpu_kernel(iter, [](auto x) { return -x; });
        }
    }

    void UnaryExp::cpu(TensorIterator& iter) {
        ScalarType dtype = iter.get_common_dtype();

        DISPATCH_FLOAT_TYPES(dtype, "unary_exp", [&] {
            kernel::unary_cpu_kernel(iter, [](scalar_t x) { return std::exp(x); });
        });
    }

    
    void UnaryLog::cpu(TensorIterator& iter) {
        ScalarType dtype = iter.get_common_dtype();

        DISPATCH_FLOAT_TYPES(dtype, "unary_log", [&] {
            kernel::unary_cpu_kernel(iter, [](scalar_t x) {
                return std::log(x);
            });
        });
    }

} // namespace tensor::ops