#include "core/TensorImpl.hpp"
#include "core/TensorIterator.hpp"
#include "ops/FillStorageOps.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cassert>

namespace tensor::ops
{

    template <typename T>
    __global__ void FillConstKernel(T* data, size_t n, T value)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < n)
            data[idx] = value;
    }
    
    template <typename T>
    __global__ void FillArangeKernel(T* data, size_t n, T start, T step);

    template <typename T>
    __global__ void FillLinspaceKernel(T* data, size_t n, T start, T end);

    template <typename T>
    __global__ void FillRandomUniformKernel(T* data, size_t n, T low, T high, uint64_t seed);

    template <typename T>
    __global__ void FillRandomNormalKernel(T* data, size_t n, T mean, T stddev, uint64_t seed);
    

    void FillConst::cuda(TensorIterator& iter, cudaStream_t stream, double value)
    {
        auto dtype = iter.get_common_dtype();

        DISPATCH_ALL_TYPES(dtype, "fill_const_cuda", [&] {
            scalar_t* output = iter.output_ptr<scalar_t>(0);
            size_t total_size = iter.get_outputs()[0]->get_total_size();

            if(iter.get_outputs()[0]->get_contiguous()) {
                constexpr int block_size = 256;
                int grid_size = (total_size + block_size - 1) / block_size;
                FillConstKernel<scalar_t><<<grid_size, block_size, 0, stream>>>(output, total_size, static_cast<scalar_t>(value));
            } else {
                throw std::runtime_error("FILLCONST OP CUDA non implemented for non contiguous tensors");
            }
        });
    }

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
        template void FillArange::cuda<T>(TensorIterator&, cudaStream_t, T, T); \
        template void FillLinspace::cuda<T>(TensorIterator&, cudaStream_t, T, T); \
        template void FillRandomUniform::cuda<T>(TensorIterator&, cudaStream_t, T, T, uint64_t); \
        template void FillRandomNormal::cuda<T>(TensorIterator&, cudaStream_t, T, T, uint64_t);

    INSTANTIATE_FILL_OPS(float)
    INSTANTIATE_FILL_OPS(double)
    INSTANTIATE_FILL_OPS(int32_t)
    INSTANTIATE_FILL_OPS(int64_t)
    INSTANTIATE_FILL_OPS(bool)

    #undef INSTANTIATE_FILL_OPS

} // namespace tensor::ops
