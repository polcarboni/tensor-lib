#include "core/TensorImpl.hpp"
#include "core/TensorIterator.hpp"
#include "ops/FillStorageOps.hpp"
#include <cstdint>
#include <cassert>
#include <random>


/**
 * Most of the filling operations should not be affected by contiguous. They only change the values in the buffer
 * without actually haivng to deal with the strides.
 * 
 * Strides are interesting only when using more than one operator. unary operators like these should not be affected
 * since each value depends only on the current position.
 */

namespace tensor::ops
{
    
    // TODO-fix: unaffected by the strides.
    // TODO: check the possibility to use both SIMD and regular operations, maybe add a flag in the base struct
    //       that can be activated/deactivated as wanted (or maybe better a preprocessor directive like th USE_CUDA)
    //       for allowing or not the use of SIMD operations.  

    void FillConst::cpu(TensorIterator& iter, double value)
    {
        auto dtype = iter.get_common_dtype();
        bool is_contiguous = iter.get_common_is_contiguous();
        
        size_t numel = iter.get_numel();
        size_t ndim  = iter.get_ndim();

        const std::vector<size_t>& shape = iter.get_shape();
        const std::vector<size_t>& strides = iter.get_strides(0);
        
        DISPATCH_ALL_TYPES(dtype, "fill_const", [&]{
            
            scalar_t val = static_cast<size_t>(value);
            scalar_t* output = iter.output_ptr<scalar_t>(0);
            size_t total_size = iter.get_outputs()[0]->get_total_size();

            if (is_contiguous) {
                std::fill(output, output + total_size, val);
                
            } else {
                std::vector<size_t> index(ndim, 0);
                for (size_t i = 0; i < numel; ++i) {
                    
                    // compute offset from the strides
                    size_t offset = 0;
                    for (size_t d = 0; d < ndim; ++d) {
                        offset += index[d] * strides[d];
                    }
                    output[offset] = val;

                    // indices increment with carry
                    for (int d = ndim - 1; d >= 0; d--) {
                        if (++index[d] < shape[d]) break;
                        index[d] = 0;
                    }
                }
            }
        });
    }

    void FillArange::cpu(TensorIterator& iter, double start, double step) {
     
        auto dtype = iter.get_common_dtype();
        bool is_contiguous = iter.get_common_is_contiguous();
        size_t numel = iter.get_numel();
        
        DISPATCH_ALL_TYPES(dtype, "fill_arange", [&] {
            scalar_t* output = iter.output_ptr<scalar_t>(0);
            
            scalar_t casted_start = static_cast<scalar_t>(start);
            scalar_t casted_step  = static_cast<scalar_t>(step);

            if (is_contiguous) {
                for (size_t i = 0; i < numel; ++i) {
                    output[i] = casted_start + casted_step * static_cast<scalar_t>(i);
                }
            } 
            
            else {    
                
                size_t ndim  = iter.get_ndim();
                const std::vector<size_t>& shape = iter.get_shape();
                const std::vector<size_t>& strides = iter.get_strides(0);
                
                std::vector<size_t> index(ndim, 0);
                
                // compute offset from the strides
                for (size_t i = 0; i < numel; ++i) {
                    size_t offset = 0;
                    for (size_t d = 0; d < ndim; ++d) {
                        offset += index[d] * strides[d];
                    }
                    output[offset] = casted_start + casted_step * static_cast<scalar_t>(i);

                    // indices increment with carry
                    for (int d = ndim - 1; d >= 0; d--) {
                        if (++index[d] < shape[d]) break;
                        index[d] = 0;
                    }
                }
            }
        });
    }

    void FillLinspace::cpu(TensorIterator& iter, double start, double end)
    {
        auto dtype = iter.get_common_dtype();
        bool is_contiguous = iter.get_common_is_contiguous();
        size_t numel = iter.get_numel();
        size_t ndim  = iter.get_ndim();

        const std::vector<size_t>& shape   = iter.get_shape();
        const std::vector<size_t>& strides = iter.get_strides(0);

        // Integer values not supported
        DISPATCH_FLOAT_TYPES(dtype, "fill_linspace", [&] {
            scalar_t* output = iter.output_ptr<scalar_t>(0);
            
            scalar_t casted_start = static_cast<scalar_t>(start);
            scalar_t casted_end   = static_cast<scalar_t>(end);

            // single element tensors have a null step
            scalar_t step = (numel > 1)
                ? (casted_end - casted_start) / static_cast<scalar_t>(numel - 1)
                :  static_cast<scalar_t>(0);

            if (is_contiguous) {
                for (size_t i = 0; i < numel; ++i) {
                    output[i] = (i == numel - 1)
                        ? casted_end
                        : casted_start + step * static_cast<scalar_t>(i);
                }
            } else {
                // Non contiguous case
                std::vector<size_t> index(ndim, 0);

                for (size_t i = 0; i < numel; ++i) {
                    size_t offset = 0;
                    for (size_t d = 0; d < ndim; ++d) {
                        offset += index[d] * strides[d];
                    }
    
                    output[offset] = (i == numel - 1)
                        ? casted_end
                        : casted_start + step * static_cast<scalar_t>(i);

                    for (int d = ndim - 1; d >= 0; d--) {
                        if (++index[d] < shape[d]) break;
                        index[d] = 0;
                    }
                }
            }
        });
    }

    void FillRandomUniform::cpu(TensorIterator& iter, double low, double high, uint64_t seed) {
        
        auto dtype = iter.get_common_dtype();
        bool is_contiguous = iter.get_common_is_contiguous();
        
        size_t numel = iter.get_numel();
        size_t ndim  = iter.get_ndim();

        const std::vector<size_t>& shape   = iter.get_shape();
        const std::vector<size_t>& strides = iter.get_strides(0);

        DISPATCH_ALL_TYPES(dtype, "fill_random_uniform", [&] {
            scalar_t* output = iter.output_ptr<scalar_t>(0);
            std::mt19937_64 gen(seed);

            // filling logic
            auto fill_tensor = [&](auto&& distribution) {
                if (is_contiguous) {
                    for (size_t i = 0; i < numel; ++i) {
                        output[i] = distribution(gen);
                    }
                } else {
                    std::vector<size_t> index(ndim, 0);
                    for (size_t i = 0; i < numel; ++i) {
                        size_t offset = 0;
                        for (size_t d = 0; d < ndim; ++d) {
                            offset += index[d] * strides[d];
                        }
                        output[offset] = distribution(gen);
                        for (int d = ndim - 1; d >= 0; d--) {
                            if (++index[d] < shape[d]) break;
                            index[d] = 0;
                        }
                    }
                }
            };

            // lambda only allows the instantiation of the correct typed branch
            [&]<typename T = scalar_t>() {
                if constexpr (std::is_floating_point_v<T>) {
                    std::uniform_real_distribution<T> dis(low,high);
                    fill_tensor(dis);
                }
                else if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
                    std::uniform_int_distribution<T> dis(static_cast<T>(low),
                                                                static_cast<T>(high));
                    fill_tensor(dis);
                }
                else if constexpr (std::is_same_v<T, bool>) {
                    std::bernoulli_distribution dis(0.5);
                    fill_tensor(dis);
                }
            }();
        });
    }

    void FillRandomNormal::cpu(TensorIterator& iter, double mean, double stddev, uint64_t seed) {
        auto dtype = iter.get_common_dtype();
        bool is_contiguous = iter.get_common_is_contiguous();
        
        size_t numel = iter.get_numel();
        size_t ndim = iter.get_ndim();
        
        const std::vector<size_t>& shape = iter.get_shape();
        const std::vector<size_t>& strides = iter.get_strides(0);

        DISPATCH_FLOAT_TYPES(dtype, "fill_random_normal", [&] {
            scalar_t* output = iter.output_ptr<scalar_t>(0);
            std::mt19937_64 gen(seed);
            // std::normal_distribution<scalar_t> dis(mean, stddev);
            std::normal_distribution<double> normal_dis(mean, stddev);

            if (is_contiguous) {
                for (size_t i = 0; i < numel; ++i) {
                    output[i] = static_cast<scalar_t>(normal_dis(gen));
                }
            } else {
                std::vector<size_t> index(ndim, 0);
                for (size_t i = 0; i < numel; ++i) {
                    size_t offset = 0;
                    for (size_t d = 0; d < ndim; ++d) {
                        offset += index[d] * strides[d];
                    }
                    output[offset] = static_cast<scalar_t>(normal_dis(gen));
                    for (int d = ndim - 1; d >= 0; d--) {
                        if (++index[d] < shape[d]) break;
                        index[d] = 0;
                    }
                }
            }
        });
    }

    // Assumes the shape is correctly squared
    void FillEye::cpu(TensorIterator& iter, size_t size)
    {
        FillConst fill_const;
        fill_const.cpu(iter, 0.0);

        ScalarType dtype = iter.get_common_dtype();
        bool is_contiguous = iter.get_common_is_contiguous();
        
        size_t numel = iter.get_numel();
        size_t ndim = iter.get_ndim();
        
        const std::vector<size_t>& shape = iter.get_shape();
        const std::vector<size_t>& strides = iter.get_strides(0);

        DISPATCH_ALL_TYPES(dtype, "fill_eye", [&] {
            scalar_t* output = iter.output_ptr<scalar_t>(0);

            if (is_contiguous) {
                for (size_t i = 0; i < size; ++i) {
                    output[i * (size + 1)] = static_cast<scalar_t>(1);
                }
            } else {
                for (size_t i = 0; i < size; ++i) {
                    size_t offset = 0;
                    for (size_t d = 0; d < ndim; ++d) {
                        offset += i * strides[d];
                    }
                    output[offset] = static_cast<scalar_t>(1);
                }
            }
        });
    }

} // namespace tensor::ops
