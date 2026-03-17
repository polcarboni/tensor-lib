#include "core/TensorImpl.hpp"
#include "core/TensorIterator.hpp"
#include "ops/UnaryOps.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cassert>

namespace tensor::ops::kernel
{

    template <typename SrcType, typename DstType>
    __global__ void cast_kernel(const SrcType* input, DstType* output, size_t numel)
    {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < numel) {
            output[idx] = static_cast<DstType>(input[idx]);
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

} // namespace tensor::ops