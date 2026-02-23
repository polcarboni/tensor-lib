#pragma once
#include "core/TensorIterator.hpp"
#include <algorithm>
#include <cstdint>

// Avoids inclusion of CUDA headers
typedef struct CUstream_st* cudaStream_t;

namespace tensor::ops
{
    
    template <Direction Dir>
    class BinaryOpBase {
    protected:
        static constexpr bool supports_broadcasting_ = true;
        static constexpr int num_inputs_ = 2;
        static constexpr int num_outputs_ = 1;
        static constexpr IterationKind iter_kind_ = IterationKind::ELEMENT_WISE;

    public:
        static constexpr bool supports_broadcasting() { return supports_broadcasting_; }
        static constexpr int num_inputs() { return num_inputs_; }
        static constexpr int num_outputs() { return num_outputs_; }
        static constexpr IterationKind iter_kind() { return iter_kind_; }
        static constexpr Direction get_direction() { return Dir; }
    };

    struct BinaryAdd : BinaryOpBase<Direction::FORWARD> {
        static void cpu(TensorIterator& iter);
        static void cuda(TensorIterator& iter, cudaStream_t stream);
    };

    struct BinaryAddBackward : BinaryOpBase<Direction::BACKWARD> {        
        static void cpu(TensorIterator& iter);
        static void cuda(TensorIterator& iter, cudaStream_t stream);
    };

    struct BinarySub : BinaryOpBase<Direction::FORWARD> {
        static void cpu(TensorIterator& iter);
        static void cuda(TensorIterator& iter, cudaStream_t stream);
    };

    struct BinaryExp : BinaryOpBase<Direction::FORWARD> {
        static void cpu(TensorIterator& iter);
        static void cuda(TensorIterator& iter, cudaStream_t stream);
    };
    
} // namespace tensor::ops