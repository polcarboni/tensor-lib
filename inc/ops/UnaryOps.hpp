#pragma once
#include "core/TensorIterator.hpp"
#include <cstdint>

typedef struct CUstream_st* cudaStream_t;

namespace tensor::ops
{
    template <Direction Dir>
    class UnaryOpBase {
    protected:
        static constexpr bool supports_broadcasting_ = false;
        static constexpr int num_inputs_ = 1;
        static constexpr int num_outputs_ = 1;
        static constexpr IterationKind iter_kind_ = IterationKind::ELEMENT_WISE;

    public:
        static constexpr bool supports_broadcasting() { return supports_broadcasting_; }
        static constexpr int num_inputs() { return num_inputs_; }
        static constexpr int num_outputs() { return num_outputs_; }
        static constexpr IterationKind iter_kind() { return iter_kind_; }
        static constexpr Direction get_direction() { return Dir; }        
    };
    
    struct UnaryCastOp : UnaryOpBase<Direction::FORWARD> {
        static void cpu(TensorIterator& iter);
        static void cuda(TensorIterator& iter, cudaStream_t stream);
    };

    /* Negation */
    struct UnaryNeg : UnaryOpBase<Direction::FORWARD> {
        static void cpu(TensorIterator& iter);
        static void cuda(TensorIterator& iter, cudaStream_t stream);
    };

    /* Make the tensor contiguous*/
    struct ContiguousOp : UnaryOpBase<Direction::FORWARD> {
        static void cpu(TensorIterator& iter);
        static void cuda(TensorIterator& iter, cudaStream_t stream);
    };

    struct UnaryNegBackward : UnaryOpBase<Direction::BACKWARD> {
        static void cpu(TensorIterator& iter);
        static void cuda(TensorIterator& iter, cudaStream_t stream);
    };

    /* Absolute value */
    struct UnaryAbs : UnaryOpBase<Direction::FORWARD> {
        static void cpu(TensorIterator& iter);
        static void cuda(TensorIterator& iter, cudaStream_t stream);
    };

    struct UnaryAbsBackward : UnaryOpBase<Direction::BACKWARD> {
        static void cpu(TensorIterator& iter);
        static void cuda(TensorIterator& iter, cudaStream_t stream);
    };

    /* Exponential */
    struct UnaryExp : UnaryOpBase<Direction::FORWARD> {
        static void cpu(TensorIterator& iter);
        static void cuda(TensorIterator& iter, cudaStream_t stream);
    };

    struct UnaryExpBackward : UnaryOpBase<Direction::BACKWARD> {
        static void cpu(TensorIterator& iter);
        static void cuda(TensorIterator& iter, cudaStream_t stream);
    };

    /* Logarithm */
    struct UnaryLog : UnaryOpBase<Direction::FORWARD> {
        static void cpu(TensorIterator& iter);
        static void cuda(TensorIterator& iter, cudaStream_t stream);
    };

    // ----------------------- ACTIVATION FUNCTIONS ----------------------- 

    struct UnarySigmoid : UnaryOpBase<Direction::FORWARD> {
        static void cpu(TensorIterator& iter);
        static void cuda(TensorIterator& iter, cudaStream_t stream);
    };

    struct UnaryTanh : UnaryOpBase<Direction::FORWARD> {
        static void cpu(TensorIterator& iter);
        static void cuda(TensorIterator& iter, cudaStream_t stream);
    };

    struct UnaryRelu : UnaryOpBase<Direction::FORWARD> {
        static void cpu(TensorIterator& iter);
        static void cuda(TensorIterator& iter, cudaStream_t stream);
    };

} // namespace tensor::ops