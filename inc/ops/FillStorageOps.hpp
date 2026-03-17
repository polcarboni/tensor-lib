#pragma once
#include "core/TensorIterator.hpp"
#include <algorithm>
#include <cstdint>
#include <random>

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
        // TODO: should these be defined in .cpp (static?). I think so.
        static constexpr bool supports_broadcasting() { return supports_broadcasting_; }
        static constexpr int num_inputs() { return num_inputs_; }
        static constexpr int num_outputs() { return num_outputs_; }
        static constexpr IterationKind iter_kind() { return iter_kind_; }
        static constexpr Direction get_direction() { return direction_; }
    };

    struct FillConst : FillOpBase {
        static void cpu(TensorIterator& iter, double value);
        static void cuda(TensorIterator& iter, cudaStream_t stream = nullptr, double value = 0);
    };
    
    struct FillBuffer : FillOpBase  {
        static void cpu(TensorIterator& iter, const double* src);
        static void cuda(TensorIterator& iter, cudaStream_t stream, const double* src);
    };  

    struct FillArange : FillOpBase  {
        static void cpu(TensorIterator& iter, double start = 0.0, double step = 1.0);
        static void cuda(TensorIterator& iter, cudaStream_t stream, double start, double step);
    };

    struct FillLinspace : FillOpBase  {
        static void cpu(TensorIterator& iter, double start = 0.0, double end = 1.0);
        static void cuda(TensorIterator& iter, cudaStream_t stream = nullptr, double start = 0.0, double end = 1.0);
    };

    struct FillRandomUniform : FillOpBase  {
        static void cpu(TensorIterator& iter, double low = 0.0, double high = 1.0, uint64_t seed = 42);
        static void cuda(TensorIterator& iter, cudaStream_t stream = nullptr, double low = 0.0, double high = 1.0, uint64_t seed = 0);
    };

    struct FillRandomNormal : FillOpBase  {
        static void cpu(TensorIterator& iter, double mean = 0.0, double stddev = 1.0, uint64_t seed = 42);
        static void cuda(TensorIterator& iter, cudaStream_t stream = nullptr, double mean = 0.0, double stddev = 1.0, uint64_t seed = 0);
    };

    struct FillEye : FillOpBase  {
        static void cpu(TensorIterator& iter, size_t size = 1);
        static void cuda(TensorIterator& iter, cudaStream_t stream = nullptr, size_t size = 1);
    };

} // namespace tensor::ops
