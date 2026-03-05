#include "core/TensorImpl.hpp"
#include "core/TensorIterator.hpp"
#include "ops/ReductionOps.hpp"
#include <cstdint>
#include <cassert>

namespace tensor::ops::kernel {

    template <typename Op>
    void reduction_cpu_kernel(TensorIterator& iter, Op op)
    {
        // placeholder
    }

} // namespace tensor::ops::kernel


namespace tensor::ops {

    void ReduceSum::cpu(TensorIterator& iter)
    {
        // placeholder
    }

} // namespace tensor::ops