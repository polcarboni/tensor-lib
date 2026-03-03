#include "core/TensorImpl.hpp"
#include "core/TensorIterator.hpp"
#include "ops/BinaryOps.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cassert>

namespace tensor::ops::kernel
{

    /**
     * TODO: required a path for non-fully reduced contiguous tensors.
     * TODO: the 1D non contiguous path might not require a separate implemenation
     *       (and use the general one instead). Maybe not, currently looks quite nice.
     */

    template <typename scalar_t, typename Op>
    __global__ void binary_cuda_kernel_impl_contiguous(scalar_t* output,
                                                       const scalar_t* lhs,
                                                       const scalar_t* rhs,
                                                       size_t numel,
                                                       Op op)
    {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < numel) {
            output[i] = op(lhs[i], rhs[i]);
        }
    }
    
    template <typename scalar_t, typename Op>
    __global__ void binary_cuda_kernel_impl_1d(scalar_t* output,
                                               const scalar_t* lhs,
                                               const scalar_t* rhs,
                                               size_t numel,
                                               size_t out_stride,
                                               size_t lhs_stride,
                                               size_t rhs_stride,
                                               Op op)
    {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < numel) {
            output[i * out_stride] = op(lhs[i * lhs_stride], rhs[i * rhs_stride]);
        }

    }

    /**
     * TODO: This guards the fixed size as the shape vectors must be copied
     *       from host to device in order to be used for them to be used by the kernel. 
     * 
     * Check again for a better description of the problem.
     */
    static constexpr size_t MAX_DIMS = 8;

    template <typename scalar_t, typename Op>
    __global__ void binary_cuda_kernel_impl_nd(scalar_t* output,
                                               const scalar_t* lhs,
                                               const scalar_t* rhs,
                                               size_t numel,
                                               size_t ndim,
                                               size_t shape[MAX_DIMS],
                                               size_t out_strides[MAX_DIMS],
                                               size_t lhs_strides[MAX_DIMS],
                                               size_t rhs_strides[MAX_DIMS],
                                               Op op)
    {

        /**
         * In this kernel, the flat idx must be converted to the actual memory offset
         * for the 3 involved operands. (With potentially 3 different strides layout).
         */
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (idx >= numel) return;

        size_t remaining  = idx;
        size_t out_offset = 0;
        size_t lhs_offset = 0;
        size_t rhs_offset = 0;


        size_t stride_map[MAX_DIMS];        // represents what strides would be if tensor were contiguous
                                            // or: how many flat indices separates 2 adjacent elements along each dimension
        stride_map[ndim - 1] = 1;           // last dimension stride is always 1
        for (int d = static_cast<int>(ndim) - 2; d >= 0; --d) {
            stride_map[d] = stride_map[d + 1] * shape[d + 1];
        }

        // For each element(thread), this loops over all the strides array of
        // operands and compute the actual physical offset that needs to be computed

        for (size_t d = 0; d < ndim; ++d) {
            size_t coord = remaining / stride_map[d];
            remaining   -= coord * stride_map[d];
            lhs_offset  += coord * lhs_strides[d];
            rhs_offset  += coord * rhs_strides[d];
            out_offset  += coord * out_strides[d];
        }

        output[out_offset] = op(lhs[lhs_offset], rhs[rhs_offset]);
    }
    
    template <typename Op>
    void binary_cuda_kernel(TensorIterator& iter, Op op, cudaStream_t stream)
    {
        auto dtype = iter.get_common_dtype();

        DISPATCH_ALL_TYPES(dtype, "binary_cuda_kernel", ([&] {
            scalar_t*       out = iter.output_ptr<scalar_t>(0);
            const scalar_t* lhs = iter.input_ptr<scalar_t>(0);
            const scalar_t* rhs = iter.input_ptr<scalar_t>(0);

            const size_t numel = iter.get_numel();
            const size_t ndim = iter.get_ndim();
            
            const auto& shape = iter.get_shape();
            const auto& lhs_strides = iter.get_strides(0);
            const auto& rhs_strides = iter.get_strides(1);
            const auto& out_strides = iter.get_strides(2);

            /**
             * TODO: is required some check on the size of input being multiple or not
             *       of the GRID size? Padding of threads?
             */
            constexpr int BLOCK = 256;
            const int GRID = static_cast<int>((numel + BLOCK - 1) / BLOCK); 

            if (iter.get_common_is_contiguous() && ndim == 1) {
                /* Fully contiguous and single dimension (coalesced) path */
                binary_cuda_kernel_impl_contiguous<scalar_t><<<GRID, BLOCK, 0, stream>>>(out, lhs, rhs, numel, op);
            }

            else if (ndim == 1) {
                /* Non contiguous 1d path */
                binary_cuda_kernel_impl_1d<scalar_t><<<GRID, BLOCK, 0, stream>>>(out, lhs, rhs, numel,
                                                                                 out_strides[0],
                                                                                 lhs_strides[0],
                                                                                 rhs_strides[0],
                                                                                 op);
            } else {
                /* General case: non contiguous, multiple dimensions */
                assert(ndim <= MAX_DIMS && "Tensor Rank exceeds MAX_DIMS");

                // Copy arrays to device memory in order for the kernel to allow read.

                size_t h_shape[MAX_DIMS]       = {};
                size_t h_lhs_strides[MAX_DIMS] = {};
                size_t h_rhs_strides[MAX_DIMS] = {};
                size_t h_out_strides[MAX_DIMS] = {};

                for (size_t d = 0; d < ndim; ++d) {
                    h_shape[d]       = shape[d];
                    h_lhs_strides[d] = lhs_strides[d];
                    h_rhs_strides[d] = rhs_strides[d];
                    h_out_strides[d] = out_strides[d];
                }


                /**
                 * TODO: why the use of Async? I assume they correct order is enforced by 
                 *       the presence of the kernel call. Varify that and provide a better
                 *       comment if required. 
                 * 
                 * TODO: Does cudaMallocAsync guarantee the memory is empty? since it allocates more then
                 *       what is possibly used it could provide problems.
                 */
                size_t *d_shape, *d_lhs_strides, *d_rhs_strides, *d_out_strides;

                cudaMallocAsync(&d_shape, MAX_DIMS * sizeof(size_t), stream);
                cudaMallocAsync(&d_lhs_strides, MAX_DIMS * sizeof(size_t), stream);
                cudaMallocAsync(&d_rhs_strides, MAX_DIMS * sizeof(size_t), stream);
                cudaMallocAsync(&d_out_strides, MAX_DIMS * sizeof(size_t), stream);

                cudaMemcpyAsync(d_shape,       h_shape,       MAX_DIMS * sizeof(size_t), cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_lhs_strides, h_lhs_strides, MAX_DIMS * sizeof(size_t), cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_rhs_strides, h_rhs_strides, MAX_DIMS * sizeof(size_t), cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_out_strides, h_out_strides, MAX_DIMS * sizeof(size_t), cudaMemcpyHostToDevice, stream);

                binary_cuda_kernel_impl_nd<scalar_t><<<GRID, BLOCK, 0, stream>>>(out, lhs, rhs, numel, ndim,
                                                                              d_shape,
                                                                              d_out_strides,
                                                                              d_lhs_strides,
                                                                              d_rhs_strides,
                                                                              op);      

                cudaFreeAsync(d_shape,       stream);
                cudaFreeAsync(d_lhs_strides, stream);
                cudaFreeAsync(d_rhs_strides, stream);
                cudaFreeAsync(d_out_strides, stream);
            }
        }));
    }

    
} // namespace tensor::ops::kernel

namespace tensor::ops
{
    // /**
    //  * Kernel signatures are provided by LLM: they might need to change the args.
    //  */

    // template <typename T>
    // __global__ void BinaryAddKernel(T* output, const T* lhs, const T* rhs, size_t n) { }
    
    // template <typename T>
    // __global__ void BinaryAddBackwardKernel(T* grad_lhs, T* grad_rhs, const T* grad_output, size_t n) { }
    
    // template <typename T>
    // __global__ void BinarySubKernel(T* output, const T* lhs, const T* rhs, size_t n) { }
    
    // template <typename T>
    // __global__ void BinaryExpKernel(T* output, const T* lhs, const T* rhs, size_t n) { }
    

    void BinaryAdd::cuda(TensorIterator& iter, cudaStream_t stream) {
        kernel::binary_cuda_kernel(iter, [] __device__ (auto a, auto b) { return a + b; }, stream);
    }
    
    void BinaryAddBackward::cuda(TensorIterator& iter, cudaStream_t stream) { }
    
    void BinarySub::cuda(TensorIterator& iter, cudaStream_t stream) { }
    
    void BinaryExp::cuda(TensorIterator& iter, cudaStream_t stream) { }

} // namespace tensor::ops