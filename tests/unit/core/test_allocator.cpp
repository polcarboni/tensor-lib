#include <catch2/catch_test_macros.hpp>
#include "core/Allocator.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace tensor;

// -------------------------------------------------------------------------------------------------------------
//                                                  CPU ALLOCATOR
// -------------------------------------------------------------------------------------------------------------

TEST_CASE("CPU Allocator allocation", "[allocator][cpu]")
{
    CPUAllocator alloc;

    SECTION("allocate returns non-null por positive sizes")
    {
        void* p = alloc.allocate(1);
        CHECK(p != nullptr);
        alloc.deallocate(p);

        p = alloc.allocate(64);
        CHECK(p != nullptr);
        alloc.deallocate(p);

        p = alloc.allocate(1024);
        CHECK(p != nullptr);
        alloc.deallocate(p);
    }

    SECTION("deallocate nullptr is no-op")
    {
        CHECK_NOTHROW(alloc.deallocate(nullptr));
    }

    SECTION("allocate returns nullptr for zero bytes")
    {
        void* p = alloc.allocate(0);
        CHECK(p == nullptr);
        alloc.deallocate(p);
    }

    SECTION("allocated memory is writeable and readable")
    {
        constexpr size_t N = 256;
        void* p = alloc.allocate(N);
        REQUIRE(p != nullptr);

        std::memset(p, 0xAB, N);
        const auto* bytes = static_cast<unsigned char*>(p);
        bool all_match = true;
        for (size_t i = 0; i < N; ++i) {
            all_match &= (bytes[i] == 0xAB);
        }
        CHECK(all_match);

        alloc.deallocate(p);
    }
}

TEST_CASE("CPU Allocator SIMD alignment", "[allocator][cpu]")
{
    CPUAllocator alloc;

    SECTION("alignment constant is 64 bytes")
    {
        CHECK(CPUAllocator::ALIGNMENT == 64);
    }

    SECTION("returned pointer is alingned to ALIGNMENT bytes for various sizes")
    {
        for (size_t n : {1u, 63u, 64u, 65u, 128u, 512u, 4096u}) {
            void* p = alloc.allocate(n);
            REQUIRE(p != nullptr);
            CHECK(reinterpret_cast<uintptr_t>(p) % CPUAllocator::ALIGNMENT == 0);
        }
    }
}

TEST_CASE("CPU Allocator multiple independent allocations", "[allocator][cpu]")
{
    CPUAllocator alloc;

    SECTION("multiple live allocations are distinct")
    {
        constexpr size_t N = 3;
        constexpr size_t SIZE = 128;
        void* ptrs[N];

        for (size_t i = 0; i < N; ++i) {
            ptrs[i] = alloc.allocate(SIZE);
            REQUIRE(ptrs[i] != nullptr);
        }

        // Distinct pointers
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = i + 1; j < N; ++j) {
                CHECK(ptrs[i] != ptrs[j]);
            }
        }

        // Independent writing to blocks
        for (size_t i = 0; i < N; ++i) {
            std::memset(ptrs[i], static_cast<int>(i + 1), SIZE);
        }

        for (size_t i = 0; i < N; ++i) {
            const auto* bytes = static_cast<unsigned char*>(ptrs[i]);
            for (size_t b = 0; b < SIZE; ++b) {
                CHECK(bytes[b] == static_cast<unsigned char>(i + 1));
            }
        }

        for (size_t i = 0; i < N; ++i) {
            alloc.deallocate(ptrs[i]);
        }
    }

    SECTION("allocate after deallocate succeeds")
    {
        void* p = alloc.allocate(64);
        REQUIRE(p != nullptr);
        alloc.deallocate(p);

        void* p2 = alloc.allocate(64);
        CHECK(p2 != nullptr);
        alloc.deallocate(p2);
    }
}

// -------------------------------------------------------------------------------------------------------------
//                                             CUDA ALLOCATOR
// -------------------------------------------------------------------------------------------------------------

TEST_CASE("CUDA Allocator without CUDA build", "[allocator][cuda]")
{
    #ifdef USE_CUDA
        SKIP("CUDA is enabled - skipping");
    #else
        CUDAAllocator alloc;

        SECTION("allocate throws std::runtime_error when CUDA is not compiled")
        {
            CHECK_THROWS_AS(alloc.allocate(128), std::runtime_error);
        }

        SECTION("allocate with zero bytes throws std::runtime_error")
        {
            CHECK_THROWS_AS(alloc.allocate(0), std::runtime_error);
        }

        SECTION("deallocate throws std::runtime_error when CUDA is not compiled")
        {
            void* ptr = reinterpret_cast<void*>(0x1);
            CHECK_THROWS(alloc.deallocate(ptr));
        }
    #endif
}

#ifdef USE_CUDA
    TEST_CASE("mock", "[allocator]")
    {
        auto alloc = CUDAAllocator();

        // This should just mirror the previous one, the CUDA logic is
        // handled within (no cuda code here).


        // -------------------------------------------------------  

        SECTION("multiple live allocations are distinct")
        {
            void* p1 = alloc.allocate(128);
            void* p2 = alloc.allocate(128);
            REQUIRE(p1 != nullptr);
            REQUIRE(p2 != nullptr);
            CHECK(p1 != p2);
            alloc.deallocate(p1);
            alloc.deallocate(p2);
        }

        SECTION("allocate after deallocate succeeds")
        {
            void* p = alloc.allocate(64);
            REQUIRE(p != nullptr);
            alloc.deallocate(p);

            void* p2 = alloc.allocate(64);
            CHECK(p2 != nullptr);
            alloc.deallocate(p2);
        }
            
    }
#endif


// TEST_CASE("CUDAAllocator without CUDA build", "[allocator][cuda]")
// {
// #ifdef USE_CUDA
//     SKIP("CUDA is enabled — skipping no-CUDA guard tests");
// #else
//     CUDAAllocator alloc;

//     SECTION("allocate throws std::runtime_error when CUDA not compiled in")
//     {
//         CHECK_THROWS_AS(alloc.allocate(128), std::runtime_error);
//     }

//     SECTION("allocate with zero bytes also throws (guard fires before early-out)")
//     {
//         CHECK_THROWS_AS(alloc.allocate(0), std::runtime_error);
//     }

//     SECTION("deallocate throws std::runtime_error when CUDA not compiled in")
//     {
//         void* fake = reinterpret_cast<void*>(0x1);
//         CHECK_THROWS(alloc.deallocate(fake));
//     }
// #endif
// }

// #ifdef USE_CUDA
// TEST_CASE("CUDAAllocator basic allocation", "[allocator][cuda]")
// {
//     CUDAAllocator alloc;

//     SECTION("allocate returns non-null for positive sizes")
//     {
//         void* p = alloc.allocate(256);
//         CHECK(p != nullptr);
//         alloc.deallocate(p);
//     }

//     SECTION("allocate returns nullptr for zero bytes")
//     {
//         void* p = alloc.allocate(0);
//         CHECK(p == nullptr);
//         CHECK_NOTHROW(alloc.deallocate(p));
//     }

//     SECTION("deallocate nullptr is a no-op")
//     {
//         CHECK_NOTHROW(alloc.deallocate(nullptr));
//     }

//     SECTION("multiple live allocations are distinct")
//     {
//         void* p1 = alloc.allocate(128);
//         void* p2 = alloc.allocate(128);
//         REQUIRE(p1 != nullptr);
//         REQUIRE(p2 != nullptr);
//         CHECK(p1 != p2);
//         alloc.deallocate(p1);
//         alloc.deallocate(p2);
//     }

//     SECTION("allocate after deallocate succeeds")
//     {
//         void* p = alloc.allocate(64);
//         REQUIRE(p != nullptr);
//         alloc.deallocate(p);

//         void* p2 = alloc.allocate(64);
//         CHECK(p2 != nullptr);
//         alloc.deallocate(p2);
//     }

//     SECTION("CUDAAllocator is usable through base Allocator pointer")
//     {
//         Allocator* base = new CUDAAllocator();
//         void* p = base->allocate(64);
//         CHECK(p != nullptr);
//         base->deallocate(p);
//         CHECK_NOTHROW(delete base);
//     }
// }
// #endif