// #define USE_CUDA    //TODO-fix: define this in the build system instead of here
// #include "Tensor2scheme.hpp"
#include "inc/ops/Backend.hpp"
#include <cuda_runtime.h>

namespace tensor
{

    // -------------------------------------------------------------------------------------------------------------  
    //                                           CUDA KERNELS
    // ------------------------------------------------------------------------------------------------------------- 

    template<typename scalar_t, typename Op>
    __global__ void nullary_kernel(scalar_t* out, Op op, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = op();
        }
    }

    template<typename scalar_t, typename Op>
    __global__ void unary_kernel(scalar_t* out, const scalar_t* in, Op op, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = op(in[idx]);
        }
    }

    // -------------------------------------------------------------------------------------------------------------  
    //                                           CUDA LAUNCHERS
    // ------------------------------------------------------------------------------------------------------------- 

    template <typename scalar_t, typename Op>
    void CUDADevice::launch_nullary(TensorIterator& iter, const Op& op)
    {
        const size_t n = iter.num_elements();
        if (n == 0) return;

        scalar_t* out_ptr = static_cast<scalar_t*>(iter.data_ptr(0));

        int threads = 256;
        int blocks = (static_cast<int>(n) + threads - 1) / threads;

        nullary_kernel<scalar_t, Op><<<blocks, threads>>>(out_ptr, op, n);

        cudaError_t err = cudaGetLastError();
        if(err!= cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }

    }

    template <typename scalar_t, typename Op>
    void CUDADevice::launch_unary(TensorIterator& iter, const Op& op)
    {
        const size_t n = iter.num_elements();
        if (n == 0) return;

        scalar_t* out_ptr = static_cast<scalar_t*>(iter.data_ptr(0));
        const scalar_t* in_ptr = static_cast<const scalar_t*>(iter.data_ptr(1));

        int threads = 256;
        int blocks = (static_cast<int>(n) + threads - 1) / threads;

        unary_kernel<scalar_t, Op><<<blocks, threads>>>(out_ptr, in_ptr, op, n);
        
        // Error check
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }
    
} // namespace tensor