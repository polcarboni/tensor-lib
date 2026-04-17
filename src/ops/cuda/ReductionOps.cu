#include "core/TensorImpl.hpp"
#include "core/TensorIterator.hpp"
#include "ops/ReductionOps.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cassert>

namespace tensor::ops::kernel
{
    template <typename scalar_t, typename Op>
    __global__ void reduction_cuda_kernel_impl_x() 
    {
        // placeholder
    }

} // namespace tensor::ops::kernel

namespace tensor::ops
{
    void ReduceSum::cuda(TensorIterator& iter, cudaStream_t stream) {
        // placeholder
    }

    void ReduceMul::cuda(TensorIterator& iter, cudaStream_t stream) {
        // placeholder
    }

    void ReduceMax::cuda(TensorIterator& iter, cudaStream_t stream) {
        // placeholder
    }

    void ReduceMin::cuda(TensorIterator& iter, cudaStream_t stream) {
        // placeholder
    }
    
} // namespace tensor::ops