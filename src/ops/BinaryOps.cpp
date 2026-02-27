#include "core/TensorImpl.hpp"
#include "core/TensorIterator.hpp"
#include "ops/BinaryOps.hpp"
#include <cstdint>
#include <cassert>

namespace tensor::ops
{
    // Naive implementation (wrong): iterator still not proivdes the strides and ndims
    // for computation. The used loop is obviously wrong.
    void BinaryAdd::cpu(TensorIterator& iter) {
        
        auto dtype = iter.get_common_dtype();

        DISPATCH_ALL_TYPES(dtype, "binary_add", [&] {
            scalar_t* output    = iter.output_ptr<scalar_t>(0);
            const scalar_t* lhs = iter.input_ptr<scalar_t>(0);
            const scalar_t* rhs = iter.input_ptr<scalar_t>(1);

            size_t total_size = iter.get_outputs()[0]->get_total_size();

            if (iter.get_common_is_contiguous()) {
                for (size_t i = 0; i < total_size; ++i) {
                    output[i] = lhs[i] + rhs[i];
                }
            } else {
                throw std::runtime_error("BINARY_ADD FOR NON CONTIGUOUS STILL NOT IMPLEMENTED");
            }
        });
    }
    
    void BinaryAddBackward::cpu(TensorIterator& iter) {}
    
    void BinarySub::cpu(TensorIterator& iter) {}
    
    void BinaryExp::cpu(TensorIterator& iter) {}

} // namespace tensor::ops