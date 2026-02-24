#pragma once
#include "core/TensorIterator.hpp"
#include <cstdint>

typedef struct CUstream_st* cudaStream_t;

namespace tensor::ops
{
    template <Direction Dir>
    class CopyOpBase {
    protected:
        static constexpr bool supports_broadcasting_ = false;
        static constexpr int num_inputs_ = 1;
        static constexpr int num_outputs_ = 1;
        static constexpr IterationKind iter_kind_ = IterationKind::COPY;

    public:
        static constexpr bool supports_broadcasting() { return supports_broadcasting_; }
        static constexpr int num_inputs() { return num_inputs_; }
        static constexpr int num_outputs() { return num_outputs_; }
        static constexpr IterationKind iter_kind() { return iter_kind_; }
        static constexpr Direction get_direction() { return Dir; }        
    };

    
    /* Identity copy */
    struct IdentityCopy : CopyOpBase<Direction::FORWARD> {
        static void cpu(TensorIterator& iter);
        static void cuda(TensorIterator& iter, cudaStream_t stream);
    };

    struct IdentityCopyBackward : CopyOpBase<Direction::BACKWARD> {
        static void cpu(TensorIterator& iter);
        static void cuda(TensorIterator& iter, cudaStream_t stream);
    };

} // namespace tensor::ops