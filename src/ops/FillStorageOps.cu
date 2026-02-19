#include "FillStorageOps.hpp"
#include "TensorImpl.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace tensor::ops
{

    template <typename T>
    __global__ void FillConstKernel(T* data, size_t n, T value);
    
    template <typename T>
    __global__ void FillArangeKernel(T* data, size_t n, T start, T step);

    template <typename T>
    __global__ void FillLinspaceKernel(T* data, size_t n, T start, T end);

    template <typename T>
    __global__ void FillRandomUniformKernel(T* data, size_t n, T low, T high, uint64_t seed);

    template <typename T>
    __global__ void FillRandomNormalKernel(T* data, size_t n, T mean, T stddev, uint64_t seed);
    

    template <typename T>
    void FillConst::cuda(TensorIterator& iter, cudaStream_t stream, T value) { }

    template <typename T>
    void FillArange::cuda(TensorIterator& iter, cudaStream_t stream, T start, T step) { }

    template <typename T>
    void FillLinspace::cuda(TensorIterator& iter, cudaStream_t stream, T start, T step) { }

    template <typename T>
    void FillRandomUniform::cuda(TensorIterator& iter, cudaStream_t stream, T low, T high, uint64_t seed) {}

    template <typename T>
    void FillRandomNormal::cuda(TensorIterator& iter, cudaStream_t stream, T mean, T stddev, uint64_t seed) { }


    // Explicit instantiations

    #define INSTANTIATE_FILL_OPS(T) \
        template void FillConst::cuda<T>(TensorIterator&, cudaStream_t, T); \
        template void FillArange::cuda<T>(TensorIterator&, cudaStream_t, T, T); \
        template void FillLinspace::cuda<T>(TensorIterator&, cudaStream_t, T, T); \
        template void FillRandomUniform::cuda<T>(TensorIterator&, cudaStream_t, T, T, uint64_t); \
        template void FillRandomNormal::cuda<T>(TensorIterator&, cudaStream_t, T, T, uint64_t);

    INSTANTIATE_FILL_OPS(float)
    INSTANTIATE_FILL_OPS(double)
    INSTANTIATE_FILL_OPS(int32_t)
    INSTANTIATE_FILL_OPS(int64_t)
    INSTANTIATE_FILL_OPS(int)
    INSTANTIATE_FILL_OPS(bool)

} // namespace tensor::ops
