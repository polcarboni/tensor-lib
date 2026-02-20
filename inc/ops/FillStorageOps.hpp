#pragma once
#include "core/TensorIterator.hpp"
#include <algorithm>
#include <cstdint>

// Avoids inclusion of CUDA headers
typedef struct CUstream_st* cudaStream_t;

namespace tensor::ops
{

    class FillOpBase {
    protected:
        static constexpr bool supports_broadcasting_ = false;
        static constexpr int num_inputs_ = 1;
        static constexpr int num_outputs_ = 1;

        static constexpr IterationKind iter_kind_ = IterationKind::ELEMENT_WISE;
        static constexpr Direction direction_ = Direction::FORWARD;

    public:
        static constexpr bool supports_broadcasting() { return supports_broadcasting_; }
        static constexpr int num_inputs() { return num_inputs_; }
        static constexpr int num_outputs() { return num_outputs_; }
        static constexpr IterationKind iter_kind() { return iter_kind_; }
        static constexpr Direction get_direction() { return direction_; }
    };

    struct FillConst : FillOpBase {
        template <typename T>
        static void cpu(tensor::TensorIterator& iter, T value)
        {
            assert(get_scalar_type<T>() == iter->get_common_dtype());

            T* output = iter.output_ptr<T>(0);

            auto& outputs = iter.get_outputs();
            size_t total_size = outputs[0]->get_total_size();

            if (outputs[0]->get_contiguous()) {
                // Fast path for contiguous tensor
                std::fill(output, output + total_size, value);
            } else {
                throw std::runtime_error("FILLCONSTOP FOR NON CONTIGUOUS STILL NOT IMPLEMENTED");
            }
        }
        
        template <typename T>
        static void cuda(tensor::TensorIterator& iter, cudaStream_t stream = nullptr, T value = T{0});
    };
    
    struct FillBuffer : FillOpBase  {
        template <typename T>
        static void cpu(tensor::TensorIterator& iter, const T* src);
        
        template <typename T>
        static void cuda(tensor::TensorIterator& iter, cudaStream_t stream, const T* src);
    };

    struct FillArange : FillOpBase  {
        template <typename T>
        static void cpu(tensor::TensorIterator& iter, T start, T step);
        
        template <typename T>
        static void cuda(tensor::TensorIterator& iter, cudaStream_t stream, T start, T step);
    };

    struct FillLinspace : FillOpBase  {
        template <typename T>
        static void cpu(tensor::TensorIterator& iter, T start, T step);
        
        template <typename T>
        static void cuda(tensor::TensorIterator& iter, cudaStream_t stream = nullptr, T start = T{0}, T step = T{1});
    };

    struct FillRandomUniform : FillOpBase  {
        template <typename T>
        static void cpu(tensor::TensorIterator& iter, T low = T{0}, T high = T{1}, uint64_t seed = 0);
        
        template <typename T>
        static void cuda(tensor::TensorIterator& iter, cudaStream_t stream = nullptr, T low = T{0}, T high = T{1}, uint64_t seed = 0);
    };

    struct FillRandomNormal : FillOpBase  {
        template <typename T>
        static void cpu(tensor::TensorIterator& iter, T mean = T{0}, T stddev = T{1}, uint64_t seed = 0);
        
        template <typename T>
        static void cuda(tensor::TensorIterator& iter, cudaStream_t stream = nullptr, T mean = T{0}, T stddev = T{1}, uint64_t seed = 0);
    };

} // namespace tensor::ops
