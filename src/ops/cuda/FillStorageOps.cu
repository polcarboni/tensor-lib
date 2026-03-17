#include "core/TensorImpl.hpp"
#include "core/TensorIterator.hpp"
#include "ops/FillStorageOps.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cassert>

namespace tensor::ops::kernel
{
    // --------------------------------------------------------------------------------------
    
    template <typename T>
    __global__ void fill_const_kernel_contiguous(
        T* data,
        size_t n,
        T value)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < n)
            data[idx] = value;
    }

    template <typename T>
    __global__ void fill_const_kernel_strided(
        T* data,
        size_t n,
        T value,
        const size_t* shape,
        const size_t* strides,
        size_t ndim)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        size_t remaining = idx;
        size_t offset = 0;

        for (int d = ndim - 1; d >= 0; --d) {
            size_t coord = remaining % shape[d]; // index along dim d
            remaining /= shape[d];
            offset += coord * strides[d];
        }

        data[offset] = value;
    }
    

    // --------------------------------------------------------------------------------------
    
    template <typename T>
    __global__ void fill_arange_kernel_contiguous(T* data, size_t n, T start, T step)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = start + static_cast<T>(idx) * step;
        }
    }
    
    template <typename T>
    __global__ void fill_arange_kernel_strided(T* data, size_t n, T start, T step,
        const size_t* shape,
        const size_t* strides,
        size_t ndim)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        size_t remaining = idx;
        size_t offset = 0;

        for (int d = ndim - 1; d >= 0; --d) {
            size_t coord = remaining % shape[d];
            remaining /= shape[d];
            offset += coord * strides[d];
        }

        data[offset] = start + static_cast<T>(idx) * step;
    }
        
        
    // --------------------------------------------------------------------------------------
    
    template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
    __global__ void fill_linspace_kernel_contiguous(
        T* data,
        size_t n,
        T start,
        T end,
        T step)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        data[idx] = (idx == n - 1) ? end : start + static_cast<T>(idx) * step;
    }
    
    template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
    __global__ void fill_linspace_kernel_strided(
        T* data,
        size_t n,
        T start,
        T end,
        T step,
        const size_t* shape,
        const size_t* strides,
        size_t ndim)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        size_t remaining = idx;
        size_t offset = 0;
        for (size_t dim = ndim; dim-- > 0;)
        {
            offset += (remaining % shape[dim]) * strides[dim];
            remaining /= shape[dim];
        }

        data[offset] = (idx == n - 1) ? end : start + static_cast<T>(idx) * step;
    }
    

    // --------------------------------------------------------------------------------------

    template <typename T>
    __global__ void fill_random_uniform_kernel_contiguous(
        T* data,
        size_t n,
        T low,
        T high,
        uint64_t seed)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        curandStatePhilox4_32_10_t state;
        curand_init(seed, idx, 0, &state);

        if constexpr (std::is_same_v<T, bool>) {
            data[idx] = curand_uniform(&state) > 0.5f;
        }

        else if constexpr (std::is_floating_point_v<T>) {
            if constexpr (std::is_same_v<T, double>) {
                // float64
                data[idx] = low + static_cast<T>(high - low) * curand_uniform_double(&state);
            } else {
                // float32
                data[idx] = low + static_cast<T>(high - low) * curand_uniform(&state);
            }
        }

        else if constexpr (std::is_integral_v<T>) {
            const float u = curand_uniform(&state);
            data[idx] = low + static_cast<T>(u * (static_cast<float>(high - low) + 1.0f));
        }
    }
    

    template <typename T>
    __global__ void fill_random_uniform_kernel_strided(
        T* data,
        size_t n,
        T low,
        T high,
        uint64_t seed,
        const size_t* shape,
        const size_t* strides,
        size_t ndim)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        curandStatePhilox4_32_10_t state;
        curand_init(seed, idx, 0, &state);

        size_t remaining = idx;
        size_t offset = 0;
        for (size_t dim = ndim; dim -- > 0;)
        {
            offset += (remaining % shape[dim]) * strides[dim];
            remaining /= shape[dim];
        }

        if constexpr (std::is_same_v<T, bool>) {
            data[offset] = curand_uniform(&state) > 0.5f;
        }

        else if constexpr (std::is_floating_point_v<T>) {
            if constexpr (std::is_same_v<T, double>) {
                data[offset] = low + static_cast<T>(high - low) * curand_uniform_double(&state);
            } else {
                data[offset] = low + static_cast<T>(high - low) * curand_uniform(&state);
            }
        }

        else if constexpr (std::is_integral_v<T>) {
            const float u = curand_uniform(&state);
            data[offset] = low + static_cast<T>(u * (static_cast<float>(high - low) + 1.0f));
        }
    }

    // --------------------------------------------------------------------------------------

    template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
    __global__ void fill_random_normal_kernel_contiguous(
        T* data,
        size_t n,
        T mean,
        T stddev,
        uint64_t seed)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        curandStatePhilox4_32_10_t state;
        curand_init(seed, idx, 0, &state);

        if constexpr (std::is_same_v<T, double>) {
            // float64
            data[idx] = mean + stddev * curand_normal_double(&state);
        } else {
            // float32
            data[idx] = mean + stddev * curand_normal(&state);
        }
    }
    
    template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
    __global__ void fill_random_normal_kernel_strided(
        T* data,
        size_t n,
        T mean,
        T stddev,
        uint64_t seed,
        const size_t* shape,
        const size_t* strides,
        size_t ndim)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        curandStatePhilox4_32_10_t state;
        curand_init(seed, idx, 0, &state);

        size_t remaining = idx;
        size_t offset = 0;
        for (size_t dim = ndim; dim-- > 0;)
        {
            offset += (remaining % shape[dim]) * strides[dim];
            remaining /= shape[dim];
        }

        if constexpr (std::is_same_v<T, double>) {
            data[offset] = mean + stddev * curand_normal_double(&state);
        } else {
            data[offset] = mean + stddev * curand_normal(&state);
        }
    }

    // --------------------------------------------------------------------------------------

    template <typename T>
    __global__ void fill_eye_kernel_contiguous(
        T* data,
        size_t size)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            data[idx * (size + 1)] = static_cast<T>(1);
        } 
    }


    template <typename T>
    __global__ void fill_eye_kernel_strided(
        T* data,
        size_t size,
        const size_t* strides,
        size_t ndim)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= size) return;

        size_t offset = 0;
        for (size_t d = 0; d < ndim; ++d) {
            offset += idx * strides[d];
        }
        data[offset] = static_cast<T>(1);
    }
    
} // namespace tensor::ops::kernel

namespace tensor::ops
{
    void FillConst::cuda(TensorIterator& iter, cudaStream_t stream, double value)
    {
        ScalarType dtype = iter.get_common_dtype();
        bool is_contiguous = iter.get_common_is_contiguous();
        size_t numel = iter.get_numel();

        DISPATCH_ALL_TYPES(dtype, "fill_const_cuda", [&] {
            scalar_t* output = iter.output_ptr<scalar_t>(0);
            scalar_t casted_value = static_cast<scalar_t>(value);
            
            constexpr int BLOCK = 256;
            int GRID = (static_cast<int>(numel) + BLOCK - 1) / BLOCK;

            if (is_contiguous) {
                kernel::fill_const_kernel_contiguous<scalar_t>
                    <<<GRID, BLOCK, 0, stream>>>(output, numel, casted_value);
            } else {
                const std::vector<size_t>& shape   = iter.get_shape();
                const std::vector<size_t>& strides = iter.get_strides(0);
                size_t ndim = iter.get_ndim();

                size_t* d_shape   = nullptr;
                size_t* d_strides = nullptr;

                cudaMallocAsync(&d_shape,   ndim * sizeof(size_t), stream);
                cudaMallocAsync(&d_strides, ndim * sizeof(size_t), stream);
                cudaMemcpyAsync(d_shape,   shape.data(),   ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_strides, strides.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);

                kernel::fill_const_kernel_strided<scalar_t>
                    <<<GRID, BLOCK, 0, stream>>>(output, numel, casted_value,
                                                 d_shape, d_strides, ndim);

                cudaFreeAsync(d_shape,   stream);
                cudaFreeAsync(d_strides, stream);
            }
        });
    }

    void FillArange::cuda(TensorIterator& iter, cudaStream_t stream, double start, double step) {
        
        ScalarType dtype   = iter.get_common_dtype();
        bool is_contiguous = iter.get_common_is_contiguous();
        size_t numel       = iter.get_numel();


        DISPATCH_ALL_TYPES(dtype, "fill_arange_cuda", [&] {
            scalar_t* output = iter.output_ptr<scalar_t>(0);

            scalar_t casted_start = static_cast<scalar_t>(start);
            scalar_t casted_step  = static_cast<scalar_t>(step);

            constexpr int BLOCK = 256;
            int GRID = (static_cast<int>(numel) + BLOCK - 1) / BLOCK;

            if (is_contiguous) {
                kernel::fill_arange_kernel_contiguous<scalar_t>
                    <<<GRID, BLOCK, 0, stream>>>(output, numel,
                                                 casted_start, casted_step);
            } else {
                size_t ndim  = iter.get_ndim();
                const std::vector<size_t>& shape = iter.get_shape();
                const std::vector<size_t>& strides = iter.get_strides(0);
                
                size_t* d_shape   = nullptr;
                size_t* d_strides = nullptr;
                
                cudaMallocAsync(&d_shape,   ndim * sizeof(size_t), stream);
                cudaMallocAsync(&d_strides, ndim * sizeof(size_t), stream);
                cudaMemcpyAsync(d_shape,   shape.data(),   ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_strides, strides.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);
                
                kernel::fill_arange_kernel_strided<scalar_t>
                    <<<GRID, BLOCK, 0, stream>>>(output, numel, casted_start, casted_step,
                                                 d_shape, d_strides, ndim);

                cudaFreeAsync(d_shape,   stream);
                cudaFreeAsync(d_strides, stream);
            }
        });
    }

    void FillLinspace::cuda(TensorIterator& iter, cudaStream_t stream, double start, double end) {

        ScalarType dtype   = iter.get_common_dtype();
        bool is_contiguous = iter.get_common_is_contiguous();
        size_t numel       = iter.get_numel();

        DISPATCH_FLOAT_TYPES(dtype, "fill_linspace_cuda", [&] {
            scalar_t* output = iter.output_ptr<scalar_t>(0);

            scalar_t casted_start = static_cast<scalar_t>(start);
            scalar_t casted_end   = static_cast<scalar_t>(end);
            
            // Step non required for single element tensors
            scalar_t casted_step  = (numel > 1)
                ? static_cast<scalar_t>((end - start) / static_cast<double>(numel - 1))
                : static_cast<scalar_t>(0);
            
            constexpr int BLOCK = 256;
            int GRID = (static_cast<int>(numel) + BLOCK - 1) / BLOCK;

            if (is_contiguous) {
                kernel::fill_linspace_kernel_contiguous<scalar_t>
                    <<<GRID, BLOCK, 0, stream>>>(output, numel,
                                                 casted_start, casted_end, casted_step);
            } else {
                size_t ndim = iter.get_ndim();
                const std::vector<size_t>& shape   = iter.get_shape();
                const std::vector<size_t>& strides = iter.get_strides(0);

                size_t* d_shape   = nullptr;
                size_t* d_strides = nullptr;

                cudaMallocAsync(&d_shape,   ndim * sizeof(size_t), stream);
                cudaMallocAsync(&d_strides, ndim * sizeof(size_t), stream);
                cudaMemcpyAsync(d_shape,   shape.data(),   ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_strides, strides.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);

                kernel::fill_linspace_kernel_strided<scalar_t>
                    <<<GRID, BLOCK, 0, stream>>>(output, numel,
                                                casted_start, casted_end, casted_step,
                                                d_shape, d_strides, ndim);

                cudaFreeAsync(d_shape,   stream);
                cudaFreeAsync(d_strides, stream);
            }
        });
    }

    void FillRandomUniform::cuda(TensorIterator& iter, cudaStream_t stream, double low, double high, uint64_t seed) {

        ScalarType dtype   = iter.get_common_dtype();
        bool is_contiguous = iter.get_common_is_contiguous();
        size_t numel       = iter.get_numel();

        DISPATCH_ALL_TYPES(dtype, "fill_random_uniform_cuda", [&] {
            scalar_t* output = iter.output_ptr<scalar_t>(0);

            scalar_t casted_low  = static_cast<scalar_t>(low);
            scalar_t casted_high = static_cast<scalar_t>(high);

            constexpr int BLOCK = 256;
            int GRID = (static_cast<int>(numel) + BLOCK - 1) / BLOCK;

            if (is_contiguous) {
                kernel::fill_random_uniform_kernel_contiguous<scalar_t>
                    <<<GRID, BLOCK, 0, stream>>>(output, numel,
                                                casted_low, casted_high, seed);
            } else {
                size_t ndim  = iter.get_ndim();
                const std::vector<size_t>& shape   = iter.get_shape();
                const std::vector<size_t>& strides = iter.get_strides(0);

                size_t* d_shape   = nullptr;
                size_t* d_strides = nullptr;

                cudaMallocAsync(&d_shape,   ndim * sizeof(size_t), stream);
                cudaMallocAsync(&d_strides, ndim * sizeof(size_t), stream);
                cudaMemcpyAsync(d_shape,   shape.data(),   ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_strides, strides.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);

                kernel::fill_random_uniform_kernel_strided<scalar_t>
                    <<<GRID, BLOCK, 0, stream>>>(output, numel,
                                                casted_low, casted_high, seed,
                                                d_shape, d_strides, ndim);

                cudaFreeAsync(d_shape,   stream);
                cudaFreeAsync(d_strides, stream);
            }
        });
    }

    void FillRandomNormal::cuda(TensorIterator& iter, cudaStream_t stream, double mean, double stddev, uint64_t seed)
    {
        ScalarType dtype   = iter.get_common_dtype();
        bool is_contiguous = iter.get_common_is_contiguous();
        size_t numel       = iter.get_numel();

        DISPATCH_FLOAT_TYPES(dtype, "fill_random_normal_cuda", [&] {
            scalar_t* output = iter.output_ptr<scalar_t>(0);

            scalar_t casted_mean   = static_cast<scalar_t>(mean);
            scalar_t casted_stddev = static_cast<scalar_t>(stddev);

            constexpr int BLOCK = 256;
            int GRID = (static_cast<int>(numel) + BLOCK - 1) / BLOCK;

            if (is_contiguous) {
                kernel::fill_random_normal_kernel_contiguous<scalar_t>
                    <<<GRID, BLOCK, 0, stream>>>(output, numel,
                                                casted_mean, casted_stddev, seed);
            } else {
                size_t ndim  = iter.get_ndim();
                const std::vector<size_t>& shape   = iter.get_shape();
                const std::vector<size_t>& strides = iter.get_strides(0);

                size_t* d_shape   = nullptr;
                size_t* d_strides = nullptr;

                cudaMallocAsync(&d_shape,   ndim * sizeof(size_t), stream);
                cudaMallocAsync(&d_strides, ndim * sizeof(size_t), stream);
                cudaMemcpyAsync(d_shape,   shape.data(),   ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_strides, strides.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream);

                kernel::fill_random_normal_kernel_strided<scalar_t>
                    <<<GRID, BLOCK, 0, stream>>>(output, numel,
                                                casted_mean, casted_stddev, seed,
                                                d_shape, d_strides, ndim);

                cudaFreeAsync(d_shape,   stream);
                cudaFreeAsync(d_strides, stream);
            }
        });
    }

    // Assumes the shape is correctly squared
    void FillEye::cuda(TensorIterator& iter, cudaStream_t stream, size_t size) {
        FillConst fill_const;
        fill_const.cuda(iter, stream, 0.0);
        
        ScalarType dtype = iter.get_common_dtype();
        bool is_contiguous = iter.get_common_is_contiguous();

        DISPATCH_ALL_TYPES(dtype, "fill_eye_cuda", [&] {
            scalar_t* output = iter.output_ptr<scalar_t>(0);

            constexpr int BLOCK = 256;
            int GRID = (static_cast<int>(size) + BLOCK - 1) / BLOCK;

            if (is_contiguous) {
                kernel::fill_eye_kernel_contiguous<scalar_t>
                    <<<GRID, BLOCK, 0, stream>>>(output, size);
            } else {
                const std::vector<size_t>& strides = iter.get_strides(0);
                size_t ndim = iter.get_ndim();

                size_t* d_strides = nullptr;
                cudaMallocAsync(&d_strides, ndim * sizeof(size_t), stream);
                cudaMemcpyAsync(d_strides, strides.data(), ndim * sizeof(size_t),
                                cudaMemcpyHostToDevice, stream);

                kernel::fill_eye_kernel_strided<scalar_t>
                    <<<GRID, BLOCK, 0, stream>>>(output, size, d_strides, ndim);

                cudaFreeAsync(d_strides, stream);
            }
        });
    }

} // namespace tensor::ops
