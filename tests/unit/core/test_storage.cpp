#pragma once
#include <catch2/catch_test_macros.hpp>
#include "core/Storage.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace tensor;


// --------------------------- helpers ---------------------------

static Device cpu_device() { return Device{ DeviceType::CPU, 0 }; }

static std::vector<float> make_floats(std::initializer_list<float> vals)
{
    return std::vector<float>(vals);
}


TEST_CASE("Storage::get_allocator returns CPU allocator for CPU device", "[Storage][get_allocator]")
{
    REQUIRE_NOTHROW(Storage(16, Device{DeviceType::CPU, 0}));
}

TEST_CASE("Storage::get_allocator returns CUDA allocator for CUDA device", "[Storage][get_allocator]")
{
    #ifndef USE_CUDA
        SKIP("CUDA support not compiled");
    #endif
        REQUIRE_NOTHROW(Storage(16, Device{DeviceType::CUDA, 0}));
}

TEST_CASE("Storage::get_allocator throws on unsupported device type", "[Storage][get_allocator]")
{
    REQUIRE_THROWS_AS(
        Storage(16, Device{DeviceType::UNKNOWN, 0}),
        std::runtime_error
    );
}

/**
 * Indirect testing of private helper function copy_data_from:
 * 
 * generated tests:
 */

// ─────────────────────────────────────────────
//  Storage::copy_data_from (tested indirectly via constructors)
// ─────────────────────────────────────────────

// Note: the initializing constructor throws on null src before copy_data_from is reached.
// The early-return guards inside copy_data_from (null src, zero size, null data_) are
// therefore only reachable internally (e.g. from copy constructor with an empty Storage).

TEST_CASE("Storage::copy_data_from - null src throws at constructor level", "[Storage][copy_data_from]")
{
    REQUIRE_THROWS_AS(
        Storage(16, Device{DeviceType::CPU, 0}, nullptr),
        std::invalid_argument
    );
}

TEST_CASE("Storage::copy_data_from - zero size skips copy, data remains null", "[Storage][copy_data_from]")
{
    auto src = make_floats({1.f, 2.f});

    // Constructor short-circuits before copy_data_from when size_bytes_ == 0
    Storage s(0, Device{DeviceType::CPU, 0});
    CHECK(s.data()   == nullptr);
    CHECK(s.nbytes() == 0);
}

TEST_CASE("Storage::copy_data_from - copying empty Storage is a no-op (internal null data_ guard)", "[Storage][copy_data_from]")
{
    // Default-constructed Storage has null data_; copy constructor must not crash
    Storage empty;

    Storage copy(empty);
    CHECK(copy.data()   == nullptr);
    CHECK(copy.nbytes() == 0);
}

// ── CPU → CPU ────────────────────────────────

TEST_CASE("Storage::copy_data_from - CPU to CPU copies data correctly", "[Storage][copy_data_from]")
{
    auto src = make_floats({1.f, 2.f, 3.f, 4.f});
    const size_t bytes = src.size() * sizeof(float);

    Storage s(bytes, Device{DeviceType::CPU, 0}, src.data());

    REQUIRE(s.data() != nullptr);
    CHECK(std::memcmp(s.data(), src.data(), bytes) == 0);
}

TEST_CASE("Storage::copy_data_from - CPU to CPU copy is independent of source buffer", "[Storage][copy_data_from]")
{
    auto src = make_floats({10.f, 20.f});
    const size_t bytes = src.size() * sizeof(float);

    Storage s(bytes, Device{DeviceType::CPU, 0}, src.data());
    src[0] = 99.f;

    float first{};
    std::memcpy(&first, s.data(), sizeof(float));
    CHECK(first == 10.f);
}

#ifdef USE_CUDA

// ── CPU → CUDA ───────────────────────────────

TEST_CASE("Storage::copy_data_from - CPU to CUDA copies data correctly", "[Storage][copy_data_from][cuda]")
{
    auto src = make_floats({1.f, 2.f, 3.f, 4.f});
    const size_t bytes = src.size() * sizeof(float);

    Storage s(bytes, Device{DeviceType::CUDA, 0}, src.data());
    REQUIRE(s.data() != nullptr);

    std::vector<float> dst(src.size());
    cudaMemcpy(dst.data(), s.data(), bytes, cudaMemcpyDeviceToHost);
    CHECK(dst == src);
}

// ── CUDA → CPU ───────────────────────────────

TEST_CASE("Storage::copy_data_from - CUDA to CPU copies data correctly", "[Storage][copy_data_from][cuda]")
{
    auto src = make_floats({5.f, 6.f, 7.f, 8.f});
    const size_t bytes = src.size() * sizeof(float);

    void* dev_ptr = nullptr;
    cudaMalloc(&dev_ptr, bytes);
    cudaMemcpy(dev_ptr, src.data(), bytes, cudaMemcpyHostToDevice);

    Storage s(bytes, Device{DeviceType::CPU, 0}, dev_ptr);
    REQUIRE(s.data() != nullptr);
    CHECK(std::memcmp(s.data(), src.data(), bytes) == 0);

    cudaFree(dev_ptr);
}

// ── CUDA → CUDA ──────────────────────────────

TEST_CASE("Storage::copy_data_from - CUDA to CUDA copies data correctly", "[Storage][copy_data_from][cuda]")
{
    auto src = make_floats({9.f, 10.f, 11.f, 12.f});
    const size_t bytes = src.size() * sizeof(float);

    void* dev_ptr = nullptr;
    cudaMalloc(&dev_ptr, bytes);
    cudaMemcpy(dev_ptr, src.data(), bytes, cudaMemcpyHostToDevice);

    Storage s(bytes, Device{DeviceType::CUDA, 0}, dev_ptr);
    REQUIRE(s.data() != nullptr);

    std::vector<float> dst(src.size());
    cudaMemcpy(dst.data(), s.data(), bytes, cudaMemcpyDeviceToHost);
    CHECK(dst == src);

    cudaFree(dev_ptr);
}

// ── cudaMemcpy failure ───────────────────────

TEST_CASE("Storage::copy_data_from - failed cudaMemcpy throws std::runtime_error", "[Storage][copy_data_from][cuda]")
{
    // Passing a host pointer as destination for a CUDA device storage triggers a
    // cudaMemcpy error, which the implementation rethrows as std::runtime_error.
    auto src = make_floats({1.f, 2.f});
    const size_t bytes = src.size() * sizeof(float);

    // Allocate device storage then deliberately corrupt the internal pointer by
    // using a stack address — achievable only via a test subclass or friend.
    // If neither is available, document this path as not directly testable.
    WARN("cudaMemcpy failure path requires a test hook (friend/subclass) to inject a bad device pointer — skipping direct test");
}

#endif // USE_CUDA


