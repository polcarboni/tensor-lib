#include "core/TensorImpl.hpp"
#include "core/TensorIterator.hpp"
#include "ops/BinaryOps.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cassert>


namespace tensor::ops
{
    /**
     * Kernel signatures are provided by LLM: they might need to change the args.
     */

    template <typename T>
    __global__ void BinaryAddKernel(T* output, const T* lhs, const T* rhs, size_t n) { }
    
    template <typename T>
    __global__ void BinaryAddBackwardKernel(T* grad_lhs, T* grad_rhs, const T* grad_output, size_t n) { }
    
    template <typename T>
    __global__ void BinarySubKernel(T* output, const T* lhs, const T* rhs, size_t n) { }
    
    template <typename T>
    __global__ void BinaryExpKernel(T* output, const T* lhs, const T* rhs, size_t n) { }
    

    void BinaryAdd::cuda(TensorIterator& iter, cudaStream_t stream) { }
    
    void BinaryAddBackward::cuda(TensorIterator& iter, cudaStream_t stream) { }
    
    void BinarySub::cuda(TensorIterator& iter, cudaStream_t stream) { }
    
    void BinaryExp::cuda(TensorIterator& iter, cudaStream_t stream) { }

} // namespace tensor::ops