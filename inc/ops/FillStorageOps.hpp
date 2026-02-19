#pragma once
#include "../core/TensorIterator.hpp"
#include <algorithm>
#include <cstdint>

// Avoids inclusion of CUDA headers
typedef struct CUstream_st* cudaStream_t;

namespace tensor::ops
{

    class FillOpBase {
    protected:
        static constexpr bool supports_broadcasting_ = false;
        static constexpr int num_inputs_ = 0;
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
        static void cpu(TensorIterator& iter, T value);
        
        template <typename T>
        static void cuda(TensorIterator& iter, cudaStream_t stream = nullptr, T value = T{0});
    };
    
    struct FillBuffer : FillOpBase  {
        template <typename T>
        static void cpu(TensorIterator& iter, const T* src);
        
        template <typename T>
        static void cuda(TensorIterator& iter, cudaStream_t stream, const T* src);
    };

    struct FillArange : FillOpBase  {
        template <typename T>
        static void cpu(TensorIterator& iter, T start, T step);
        
        template <typename T>
        static void cuda(TensorIterator& iter, cudaStream_t stream, T start, T step);
    };

    struct FillLinspace : FillOpBase  {
        template <typename T>
        static void cpu(TensorIterator& iter, T start, T step);
        
        template <typename T>
        static void cuda(TensorIterator& iter, cudaStream_t stream = nullptr, T start = T{0}, T step = T{1});
    };

    struct FillRandomUniform : FillOpBase  {
        template <typename T>
        static void cpu(TensorIterator& iter, T low = T{0}, T high = T{1}, uint64_t seed = 0);
        
        template <typename T>
        static void cuda(TensorIterator& iter, cudaStream_t stream = nullptr, T low = T{0}, T high = T{1}, uint64_t seed = 0);
    };

    struct FillRandomNormal : FillOpBase  {
        template <typename T>
        static void cpu(TensorIterator& iter, T mean = T{0}, T stddev = T{1}, uint64_t seed = 0);
        
        template <typename T>
        static void cuda(TensorIterator& iter, cudaStream_t stream = nullptr, T mean = T{0}, T stddev = T{1}, uint64_t seed = 0);
    };

} // namespace tensor::ops
