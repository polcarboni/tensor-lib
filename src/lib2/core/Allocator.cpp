#include "inc/core/Allocator.hpp"
#include <cstdlib>

namespace tensor {
    
    void* CPUAllocator::allocate(size_t n) {}
    void CPUAllocator::deallocate(void* p) {}

    void* CUDAAllocator::allocate(size_t n) {}
    void CUDAAllocator::deallocate(void* p) {}

} // namespace tensor