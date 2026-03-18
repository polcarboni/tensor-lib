#pragma once
#include "core/TensorIterator.hpp"
#include <cstdint>

typedef struct CUstream_st* cudaStream_t;

namespace tensor::ops
{
    template <Direction Dir>
    class ReductionOpBase {
    protected:
        static constexpr bool supports_broadcasting_ = true;
        static constexpr int num_inputs_ = 1;
        static constexpr int num_outputs_ = 1;
        static constexpr IterationKind iter_kind_ = IterationKind::REDUCTION;

    public:
        static constexpr bool supports_broadcasting() { return supports_broadcasting_; }
        static constexpr int num_inputs()             { return num_inputs_; }
        static constexpr int num_outputs()            { return num_outputs_; }
        static constexpr IterationKind iter_kind()    { return iter_kind_; }
        static constexpr Direction get_direction()    { return Dir; }
    };


    /* Sum Reduction */
    struct ReduceSum : ReductionOpBase<Direction::FORWARD> {
        template <typename T>
        static constexpr T identity() { return static_cast<T>(0); }

        template <typename T>
        T operator()(T a, T b) const { return a + b; }
        
        static void cpu(TensorIterator& iter);
        static void cuda(TensorIterator& iter, cudaStream_t stream);
    };

    /* Reduction Backward (usually an "Expand" or "Broadcast" op) */
    struct ReduceSumBackward : ReductionOpBase<Direction::BACKWARD> {
        static void cpu(TensorIterator& iter);
        static void cuda(TensorIterator& iter, cudaStream_t stream);
    };
} // namespace tensor::ops