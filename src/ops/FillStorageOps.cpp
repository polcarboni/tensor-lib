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

    void FillConst::cpu(TensorIterator& iter, double value)
    {
        auto dtype = iter.get_common_dtype();
        
        DISPATCH_ALL_TYPES(dtype, "fill_const", [&]{
            scalar_t* output = iter.output_ptr<scalar_t>(0);
            size_t total_size = iter.get_outputs()[0]->get_total_size();
            
            // TODO-fix: transposed tensors (non contiguous) should also use this
            if (iter.get_outputs()[0]->get_contiguous()) {
                std::fill(output, output + total_size, static_cast<scalar_t>(value));
                
            } else {
                // Useful for cases of partial ownership.
                throw std::runtime_error("FILLCONSTOP FOR NON CONTIGUOUS STILL NOT IMPLEMENTED");
            }
        });
    }

    void FillArange::cpu(TensorIterator& iter, double start, double step) {}

    void FillLinspace::cpu(TensorIterator& iter, double start, double step) {}

    void FillRandomUniform::cpu(TensorIterator& iter, double low, double high, uint64_t seed) {}

    void FillRandomNormal::cpu(TensorIterator& iter, double mean, double stddev, uint64_t seed) {}

} // namespace tensor::ops
