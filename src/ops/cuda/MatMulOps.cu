#include "core/TensorImpl.hpp"
#include "core/TensorIterator.hpp"
#include "ops/MatMulOps.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cassert>

namespace tensor::ops::kernel {
}

namespace tensor::ops {
    void MatMul::cuda(TensorIterator& iter, cudaStream_t stream) {
        // placeholder
    }
}