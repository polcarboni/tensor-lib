#include "core/TensorImpl.hpp"
#include "core/TensorIterator.hpp"
#include "ops/UnaryOps.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cassert>

namespace tensor::ops::kernel
{
    enum class TypeDispatch { All, Float };

    template <typename SrcType, typename DstType>
    __global__ void cast_kernel(const SrcType* input, DstType* output, size_t numel)
    {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < numel) {
            output[idx] = static_cast<DstType>(input[idx]);
        }
    }

    template <typename scalar_t, typename Op>
    __global__ void unary_cuda_kernel_impl_contiguous(
        scalar_t* output,
        const scalar_t* input,
        size_t numel,
        Op op)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < numel) {
            output[idx] = op(input[idx]);
        }
    }

    template <typename scalar_t, typename Op>
    __global__ void unary_cuda_kernel_impl_1d(
        scalar_t* output,
        const scalar_t* input,
        size_t numel,
        size_t out_stride,
        size_t in_stride,
        Op op)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < numel) {
            output[idx * out_stride] = op(input[idx * in_stride]);
        }
    }

    static constexpr size_t MAX_DIMS = 8;

    template <typename scalar_t, typename Op>
    __global__ void unary_cuda_kernel_impl_nd(
       scalar_t* output,
       const scalar_t* input,
       size_t numel,
       size_t ndim,
       size_t shape[MAX_DIMS],
       size_t out_strides[MAX_DIMS],
       size_t in_strides[MAX_DIMS],
       Op op)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numel) return;

        size_t remaining = idx;
        size_t out_offset = 0;
        size_t in_offset = 0;

        size_t stride_map[MAX_DIMS] = {};
        stride_map[ndim - 1] = 1;
        
        for (int d = static_cast<int>(ndim) - 2; d >= 0; --d) {
            stride_map[d] = stride_map[d + 1] * shape[d + 1];
        }

        for (size_t d = 0; d < ndim; ++d) {
            size_t coord = remaining / stride_map[d];
            remaining -= coord * stride_map[d];
            in_offset += coord * in_strides[d];
            out_offset += coord * out_strides[d];
        }

        output[out_offset] = op(input[in_offset]);
    }

    template <typename scalar_t, typename Op>
    void unary_cuda_kernel_dispatch(TensorIterator& iter, Op op, cudaStream_t stream)
    {
        scalar_t* output      = iter.output_ptr<scalar_t>(0);
        const scalar_t* input = iter.input_ptr<scalar_t>(0);

        const size_t numel = iter.get_numel();
        const size_t ndim  = iter.get_ndim();

        const auto& shape       = iter.get_shape();
        const auto& in_strides  = iter.get_strides(0);
        const auto& out_strides = iter.get_strides(1);

        constexpr int BLOCK = 256;
        const int GRID = static_cast<int>((numel + BLOCK - 1) / BLOCK);

        if (iter.get_common_is_contiguous() && ndim == 1) {
            unary_cuda_kernel_impl_contiguous<scalar_t>
                <<<GRID, BLOCK, 0, stream>>>(output, input, numel, op);
        }
        else if (ndim == 1) {
            unary_cuda_kernel_impl_1d<scalar_t>
                <<<GRID, BLOCK, 0, stream>>>(output, input, numel, out_strides[0], in_strides[0], op);
        }
        else {
            assert(ndim <= MAX_DIMS);

            size_t h_shape[MAX_DIMS]          = {};
            size_t h_input_strides[MAX_DIMS]  = {};
            size_t h_output_strides[MAX_DIMS] = {};

            for (size_t d = 0; d < ndim; ++d) {
                h_shape[d]          = shape[d];
                h_input_strides[d]  = in_strides[d];
                h_output_strides[d] = out_strides[d];
            }

            size_t *d_shape, *d_input_strides, *d_output_strides;

            cudaMallocAsync(&d_shape,          MAX_DIMS * sizeof(size_t), stream);
            cudaMallocAsync(&d_input_strides,  MAX_DIMS * sizeof(size_t), stream);
            cudaMallocAsync(&d_output_strides, MAX_DIMS * sizeof(size_t), stream);

            cudaMemcpyAsync(d_shape,          h_shape,          MAX_DIMS * sizeof(size_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_input_strides,  h_input_strides,  MAX_DIMS * sizeof(size_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_output_strides, h_output_strides, MAX_DIMS * sizeof(size_t), cudaMemcpyHostToDevice, stream);

            unary_cuda_kernel_impl_nd<scalar_t><<<GRID, BLOCK, 0, stream>>>(
                output, input, numel, ndim,
                d_shape, d_output_strides, d_input_strides,
                op);

            cudaFreeAsync(d_shape,          stream);
            cudaFreeAsync(d_input_strides,  stream);
            cudaFreeAsync(d_output_strides, stream);
        }
    }

    template <TypeDispatch TD, typename Op>
    void unary_cuda_kernel(TensorIterator& iter, Op op, cudaStream_t stream) {

        ScalarType dtype = iter.get_common_dtype();

        if constexpr (TD == TypeDispatch::All) {
            DISPATCH_ALL_TYPES(dtype, "unary_cuda_kernel", [&] {
                unary_cuda_kernel_dispatch<scalar_t>(iter, op, stream);
            });
        } else {
            DISPATCH_FLOAT_TYPES(dtype, "unary_cuda_kernel", [&] {
                unary_cuda_kernel_dispatch<scalar_t>(iter, op, stream);
            });
        }
    }

} // namespace tensor::ops::kernel


namespace tensor::ops
{
    void UnaryCastOp::cuda(TensorIterator& iter, cudaStream_t stream)
    {
        auto to_dtype   = iter.get_common_dtype();
        auto from_dtype = iter.get_input_dtype();

        auto numel = iter.get_numel();

        constexpr int BLOCK = 256;
        const int GRID = static_cast<int>((numel + BLOCK - 1) / BLOCK);

        DISPATCH_ALL_TYPES(from_dtype, "castOp", [&] {
            using SrcType = scalar_t;
            const SrcType* input = iter.input_ptr<SrcType>(0);
            DISPATCH_ALL_TYPES(to_dtype, "castOp", [&] {
                using DstType = scalar_t;
                DstType* output = iter.output_ptr<DstType>(0);

                kernel::cast_kernel<SrcType, DstType><<<GRID, BLOCK, 0, stream>>>(input, output, numel);
            });
        });
    }

    void ContiguousOp::cuda(TensorIterator& iter, cudaStream_t stream) {
        // PLACEHOLDER 
    }

    void UnaryNeg::cuda(TensorIterator& iter, cudaStream_t stream) {
        kernel::unary_cuda_kernel<kernel::TypeDispatch::All>(iter, [] __device__ (auto x) {    
            if constexpr(std::is_same_v<decltype(x), bool>) {
                return !x;
            } else {
                return -x;
            }
        }, stream);
    }

    void UnaryExp::cuda(TensorIterator& iter, cudaStream_t stream) {
        kernel::unary_cuda_kernel<kernel::TypeDispatch::Float>(iter, [] __device__(auto x) {
            return exp(x);
        }, stream);
    }

    void UnaryLog::cuda(TensorIterator& iter, cudaStream_t stream) {
        kernel::unary_cuda_kernel<kernel::TypeDispatch::Float>(iter, [] __device__(auto x) {
            return log(x);
        }, stream);
    }

    // ----------------------- ACTIVATION FUNCTIONS ----------------------- 

    void UnarySigmoid::cuda(TensorIterator& iter, cudaStream_t stream) {
        ScalarType dtype = iter.get_common_dtype();

        kernel::unary_cuda_kernel<kernel::TypeDispatch::Float>(iter, [] __device__ (auto x) {
            using T = decltype(x);
            return T(1) / (T(1) + exp(-x));
        }, stream);
    }

    void UnaryTanh::cuda(TensorIterator& iter, cudaStream_t stream) {
        kernel::unary_cuda_kernel<kernel::TypeDispatch::All>(iter, [] __device__ (auto x) {
            using T = decltype(x);
            return x > T(0) ? x : T(0);
        }, stream);
    }

    void UnaryRelu::cuda(TensorIterator& iter, cudaStream_t stream) {
        kernel::unary_cuda_kernel<kernel::TypeDispatch::All>(iter, [] __device__ (auto x) {
            using T = decltype(x);
            return x > T(0) ? x : T(0);
        }, stream);
    }

} // namespace tensor::ops