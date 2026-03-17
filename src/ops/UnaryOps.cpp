#include "core/TensorImpl.hpp"
#include "core/TensorIterator.hpp"
#include "ops/UnaryOps.hpp"
#include <cstdint>
#include <cassert>
#include <vector>
#include <cstring>
#include <cmath>


namespace tensor::ops {
    
    void UnaryCastOp::cpu(TensorIterator& iter)
    {
        auto to_dtype   = iter.get_common_dtype();
        auto from_dtype = iter.get_input_dtype();

        auto numel = iter.get_numel();

        DISPATCH_ALL_TYPES(from_dtype, "castOp", [&] {
            using SrcType = scalar_t;
            scalar_t* input = iter.input_ptr<SrcType>(0);
            DISPATCH_ALL_TYPES(to_dtype, "castOp", [&] {
                using DstType = scalar_t;
                scalar_t* output = iter.output_ptr<DstType>(0);

                for (size_t i = 0; i < numel; ++i) {
                    output[i] = static_cast<DstType>(input[i]);
                }

            });
        });
    }

} // namespace tensor::ops