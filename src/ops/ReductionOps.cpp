#include "core/TensorImpl.hpp"
#include "core/TensorIterator.hpp"
#include "ops/ReductionOps.hpp"
#include <cstdint>
#include <cassert>
#include <optional>
#include <vector>

namespace tensor::ops::kernel {

    template <typename Op>
    void reduction_cpu_kernel(TensorIterator& iter, Op op)
    {
        auto dtype = iter.get_common_dtype();

        DISPATCH_ALL_TYPES(dtype, "reduction_cpu_kernel", ([&] {
            scalar_t* output      = iter.output_ptr<scalar_t>(0);
            const scalar_t* input = iter.input_ptr<scalar_t>(0);

            const auto& input_strides  = iter.get_strides(0);
            const auto& output_strides = iter.get_strides(1);

            const auto& shape = iter.get_shape();
            size_t numel      = iter.get_numel();
            size_t ndim       = iter.get_ndim();

            const auto& reduction_axes = iter.get_reduction_axes();
            const bool  keepdims       = iter.get_keepdims();

            // Compute the number of elements of the output tensors
            const auto& is_reduced_dim = iter.get_is_reduced_dim();

            size_t out_numel = 1;
            for (size_t d = 0; d < ndim; ++d) {
                if (!(*is_reduced_dim)[d]) {
                    out_numel *= shape[d];
                }
            }
            
            const scalar_t identity = op.template identity<scalar_t>();
            
            // Initialize the output with identity values
            // for (size_t i = 0; i < out_numel; ++i) {
                //     output[i] = identity;
                // }
                
                std::vector<size_t> out_counter(ndim, 0);
                size_t out_init_offset = 0;
                for (size_t i = 0; i < out_numel; ++i) {
                
                output[out_init_offset] = identity;
                for (int d = static_cast<int>(ndim)-1; d >= 0; --d) {
                    if (output_strides[d] == 0) continue;
                    out_init_offset += output_strides[d];

                    if (++out_counter[d] < shape[d]) break;
                    out_counter[d] = 0;
                    out_init_offset -= shape[d] * output_strides[d];
                }
            }
            
            if (out_numel == 1 && iter.get_common_is_contiguous()) {
                
                // Contiguous full reduction
                scalar_t acc = identity;
                for (size_t i = 0; i < numel; ++i) {
                    acc = op(acc, input[i]);
                }
                
                output[0] = acc;
                return;
            }

            else if (out_numel == 1 && !iter.get_common_is_contiguous()) {
                
                // Non contiguous full reduction
                scalar_t acc = identity;
                std::vector<size_t> counter(ndim, 0);
                size_t in_offset = 0;

                for (size_t i = 0; i < numel; ++i) {
                    acc = op(acc, input[in_offset]);

                    for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                        in_offset += input_strides[d];
                        if (++counter[d] < shape[d]) break;

                        in_offset -= shape[d] * input_strides[d];
                        counter[d] = 0;
                    }
                }
                output[0] = acc;
                return;
            }

            else if (iter.get_num_reduced_axes() == 1 && iter.get_contiguous_along_reduced_axes()) {
                
                // Single axis contiguous reduction
                const auto& is_reduced_dim = iter.get_is_reduced_dim();

                size_t reduced_dim = 0;
                size_t reduced_extent = 0;
                size_t reduced_stride = 0;

                for (size_t d = 0; d < ndim; ++d) {
                    if((*is_reduced_dim)[d]) {
                        reduced_dim = d;
                        reduced_extent = shape[d];
                        reduced_stride = input_strides[d];
                        break;
                    }
                }

                std::vector<size_t> counter(ndim, 0);
                size_t in_offset = 0;
                size_t out_offset = 0;

                for (size_t i = 0; i < out_numel; ++i) {
                    scalar_t acc     = identity;
                    size_t inner_off = in_offset;

                    for (size_t r = 0; r < reduced_extent; ++r) {
                        acc = op(acc, input[inner_off]);
                        inner_off += reduced_stride;
                    }
                    output[out_offset] = acc;

                    for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                        if((*is_reduced_dim)[d]) continue;

                        in_offset  += input_strides[d];
                        out_offset += output_strides[d];

                        if (++counter[d] < shape[d]) break;

                        counter[d] = 0;
                        in_offset  -= shape[d] * input_strides[d];
                        out_offset -= shape[d] * output_strides[d];
                    }
                }
            }

            else {

                // General path: no contiguous, n reduction axes
                std::vector<size_t> counter(ndim, 0);
                
                for (size_t i = 0; i < numel; ++i) {
                    size_t in_offset  = 0;
                    size_t out_offset = 0;
                    
                    for (int d = 0; d < static_cast<int>(ndim); ++d) {
                        in_offset += counter[d] * input_strides[d];
                        if(!(*is_reduced_dim)[d]) {
                            out_offset += counter[d] * output_strides[d];
                        }
                    }

                    output[out_offset] = op(output[out_offset], input[in_offset]);

                    for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                        ++counter[d];
                        if (counter[d] < shape[d]) break;
                        counter[d] = 0;

                    }
                }
            }
        }));
    }
} // namespace tensor::ops::kernel


namespace tensor::ops {

    void ReduceSum::cpu(TensorIterator& iter)
    {        
        kernel::reduction_cpu_kernel(iter, ReduceSum{});
    }

    void ReduceMul::cpu(TensorIterator& iter) {
        kernel::reduction_cpu_kernel(iter, ReduceMul{});
    }

    void ReduceMax::cpu(TensorIterator& iter) {
        kernel::reduction_cpu_kernel(iter, ReduceMax{});
    }

    void ReduceMin::cpu(TensorIterator& iter) {
        kernel::reduction_cpu_kernel(iter, ReduceMin{});
    }

} // namespace tensor::ops