#include "core/Allocator.hpp"
#include <cstdlib>
#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace tensor {
    
    // -------------------------------------------------------------------------------------------------------------  
    //                                                 CPU ALLOCATOR 
    // -------------------------------------------------------------------------------------------------------------  
    
    void* CPUAllocator::allocate(size_t n)
    {
        if (n == 0) return nullptr;
        void* ptr = ::operator new(n, std::align_val_t(ALIGNMENT));
        if (!ptr) throw std::bad_alloc();
        return ptr;
    }

    void CPUAllocator::deallocate(void* p)
    {
        if (!p) return;
        ::operator delete(p, std::align_val_t(ALIGNMENT));
    }
    

    
    // -------------------------------------------------------------------------------------------------------------  
    //                                                CUDA ALLOCATOR 
    // -------------------------------------------------------------------------------------------------------------  

    void* CUDAAllocator::allocate(size_t n)
    {
        #ifdef USE_CUDA
            if (n == 0) return nullptr;
            void* ptr = nullptr;
            cudaError_t = cudaMalloc(&ptr, n);
            if(err != cudaSuccess)
                throw std::runtime_error("CUDA allocation failed");
            return ptr;
        #else
            throw std::runtime_error("CUDAAllocator requires CUDA compilation");
        #endif
    }

    void CUDAAllocator::deallocate(void* p)
    {
        #ifdef USE_CUDA
        if(!p) return;
            cudaError_t err = cudaFree(p);
            if(err != cudaSuccess)
                throw std::runtime_error("CUDA deallocation failed");
        #else
            throw std::runtime_error("CUDAAllocator requires CUDA compilation");
        #endif
    }

} // namespace tensor