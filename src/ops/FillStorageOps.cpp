#include "FillStorageOps.hpp"
#include "TensorImpl.hpp"
#include <cstdint>

namespace tensor::ops
{
    template <typename T>
    void FillConst::cpu(TensorIterator& iter, T value) {}

    template <typename T>
    void FillArange::cpu(TensorIterator& iter, T start, T step) {}

    template <typename T>
    void FillLinspace::cpu(TensorIterator& iter, T start, T step) {}

    template <typename T>
    void FillRandomUniform::cpu(TensorIterator& iter, T low, T high, uint64_t seed) {}

    template <typename T>
    void FillRandomNormal::cpu(TensorIterator& iter, T mean, T stddev, uint64_t seed) {}


    // Explicit instantiations

    #define INSTANTIATE_FILL_OPS_CPU(T) \
        template void FillConst::cpu<T>(TensorIterator&, T); \
        template void FillArange::cpu<T>(TensorIterator&, T, T); \
        template void FillLinspace::cpu<T>(TensorIterator&, T, T); \
        template void FillRandomUniform::cpu<T>(TensorIterator&, T, T, uint64_t); \
        template void FillRandomNormal::cpu<T>(TensorIterator&, T, T, uint64_t);

    INSTANTIATE_FILL_OPS_CPU(float)
    INSTANTIATE_FILL_OPS_CPU(double)
    INSTANTIATE_FILL_OPS_CPU(int32_t)
    INSTANTIATE_FILL_OPS_CPU(int64_t)
    INSTANTIATE_FILL_OPS_CPU(int)
    INSTANTIATE_FILL_OPS_CPU(bool)

} // namespace tensor::ops
