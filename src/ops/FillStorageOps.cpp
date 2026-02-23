#include "core/TensorImpl.hpp"
#include "ops/FillStorageOps.hpp"
#include <cstdint>
#include <cassert>


/**
 * Most of the filling operations should not be affected by contiguous. They only change the values in the buffer
 * without actually haivng to deal with the strides.
 * 
 * Strides are interesting only when using more than one operator. unary operators like these should not be affected
 * since each value depends only on the current position.
 */

namespace tensor::ops
{
    
    // TODO-fix: unaffected by the strides.
    // TODO: check the possibility to use both SIMD and regular operations, maybe add a flag in the base struct
    //       that can be activated/deactivated as wanted (or maybe better a preprocessor directive like th USE_CUDA)
    //       for allowing or not the use of SIMD operations.  

    template <typename T>
    void FillConst::cpu(TensorIterator& iter, T value)
    {
        assert(get_scalar_type<T>() == iter.get_common_dtype());

        T* output = iter.output_ptr<T>(0);

        // auto& outputs = iter.get_outputs();
        size_t total_size = iter.get_outputs()[0]->get_total_size();

        if (iter.get_outputs()[0]->get_contiguous()) {
            // Fast path for contiguous tensor
            std::fill(output, output + total_size, value);
        } else {
            throw std::runtime_error("FILLCONSTOP FOR NON CONTIGUOUS STILL NOT IMPLEMENTED");
        }
    }

    template <typename T>
    void FillArange::cpu(TensorIterator& iter, T start, T step) {}

    template <typename T>
    void FillLinspace::cpu(TensorIterator& iter, T start, T step) {}

    template <typename T>
    void FillRandomUniform::cpu(TensorIterator& iter, T low, T high, uint64_t seed) {}

    template <typename T>
    void FillRandomNormal::cpu(TensorIterator& iter, T mean, T stddev, uint64_t seed) {}


    // Explicit instantiations

    #define INSTANTIATE_FILL_OPS_CPU(T)                                           \
        template void FillConst::cpu<T>(TensorIterator&, T);                      \
        template void FillArange::cpu<T>(TensorIterator&, T, T);                  \
        template void FillLinspace::cpu<T>(TensorIterator&, T, T);                \
        template void FillRandomUniform::cpu<T>(TensorIterator&, T, T, uint64_t); \
        template void FillRandomNormal::cpu<T>(TensorIterator&, T, T, uint64_t);

    INSTANTIATE_FILL_OPS_CPU(float)
    INSTANTIATE_FILL_OPS_CPU(double)
    INSTANTIATE_FILL_OPS_CPU(int32_t)
    INSTANTIATE_FILL_OPS_CPU(int64_t)
    INSTANTIATE_FILL_OPS_CPU(bool)

} // namespace tensor::ops
