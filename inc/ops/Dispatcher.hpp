#pragma once
#include "TensorIterator.hpp"

namespace tensor
{
    // ------------------------------------------------ factory dispatcher --------------------------------------------
    // Useful for producing tensors (ones, zeros, arange, fill_)

    template<template <typename> class Op>
    struct NullaryElementwiseDispatcher {
        static void call(TensorIterator& iter, const std::string& op_name) {
            if (iter.device().type == DeviceType::CPU) {
                DISPATCH_ALL_TYPES(iter.common_dtype(), op_name, [&] {
                    CPUDevice::launch_nullary<scalar_t, Op<scalar_t>>(iter, Op<scalar_t>{});
                });
            }
            #ifdef USE_CUDA
                else if (iter.device().type == DeviceType::CUDA) {
                    DISPATCH_ALL_TYPES(iter.common_dtype(), op_name, [&] {
                        CUDADevice::launch_nullary<scalar_t, Op<scalar_t>>(iter, Op<scalar_t>{});
                    });
                }
            #endif
            else {
                throw std::runtime_error("NullaryElementwiseDispatcher: Unsupported device type");
            }
        }
    };
    
    // ------------------------------------------------ comparison dispatcher --------------------------------------------
    // Technically binary but the output type is always bool

    template<typename Op>
    struct ComparisonDispatcher {
        static void call(TensorIterator& iter, const std::string& name);
    };

    // ------------------------------------------------ element-wise dispatcher --------------------------------------------

    /* Abstract backend dispatcher to be specialized for CPU and CUDA backends */
    template <template <typename> class Op>
    struct UnaryElementwiseDispatcher {
        static void call(TensorIterator& iter, const std::string& name);
    };

    template <template <typename> class Op>
    struct BinaryElementwiseDispatcher {
        static void call(TensorIterator& iter, const std::string& name);
    };

    template <template <typename> class Op>
    struct TernaryElementwiseDispatcher {
        static void call(TensorIterator& iter, const std::string& name);
    };


    // ----------------------------------------- reduction dispatcher --------------------------------
    
    /* Reduction requires specific iterator configurations where the output has fewer dims */
    struct ReductionDispatcher {
        template<typename scalar_t, typename acc_t>
        static void call(TensorIterator& iter, const std::string& name, acc_t identity);
    };
    
 
}