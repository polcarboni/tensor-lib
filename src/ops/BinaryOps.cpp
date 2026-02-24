#include "core/TensorImpl.hpp"
#include "core/TensorIterator.hpp"
#include "ops/BinaryOps.hpp"
#include <cstdint>
#include <cassert>

namespace tensor::ops
{
    void BinaryAdd::cpu(TensorIterator& iter) {}
    
    void BinaryAddBackward::cpu(TensorIterator& iter) {}
    
    void BinarySub::cpu(TensorIterator& iter) {}
    
    void BinaryExp::cpu(TensorIterator& iter) {}

} // namespace tensor::ops