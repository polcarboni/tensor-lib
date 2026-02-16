#pragma once
#include <cstddef>

namespace tensor {
    
    // -------------------------------------------------------------------------------------------------------------  
    //                                                  ALLOCATORS 
    // -------------------------------------------------------------------------------------------------------------  


    /* Generic allocator interface: specializes for CPU and GPU */
    class Allocator {
    public:
        virtual void* allocate(size_t num_bytes) = 0;
        virtual void deallocate(void* ptr) noexcept = 0;
        virtual ~Allocator() = default;
    };

    /* CPU allocator: SIMD aligned */
    class CPUAllocator : public Allocator
    {
    public:
        static constexpr size_t ALIGNMENT = 64;  //AVX-512/SIMD
        void* allocate(size_t n) override;
        void deallocate(void* p) noexcept override;
    };

    /* CUDA GPU allocator */
    class CUDAAllocator : public Allocator {
    public:
        void* allocate(size_t n) override;
        void deallocate(void* p) noexcept override;
    };

} // namespace tensor