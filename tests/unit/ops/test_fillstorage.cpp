#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "core/TensorImpl.hpp"
#include "core/TensorIterator.hpp"
#include "ops/FillStorageOps.hpp"

using namespace tensor;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a contiguous TensorImpl of the given shape / dtype and return it.
static std::shared_ptr<TensorImpl> make_contiguous(
    std::vector<size_t> shape, ScalarType dtype)
{
    return std::make_shared<TensorImpl>(shape, dtype); // assumes default ctor = contiguous
}

/// Build a non-contiguous (transposed) view so strides are not C-order.
// static std::shared_ptr<TensorImpl> make_non_contiguous(
//     std::vector<size_t> shape, ScalarType dtype)
// {
//     auto t = make_contiguous(shape, dtype);
//     t->transpose(0, 1);          // swap dims → non-contiguous
//     return t;
// }


TEST_CASE("fill_const fills a contiguous float tensor", "[ops][fill]")
{
    auto t = make_contiguous({2, 3}, ScalarType::Float32);
    t->fill_const(2.5);

    const float* p = t->data_ptr<float>();
    for (size_t i = 0; i < t->total_size_; ++i)
        CHECK(p[i] == Catch::Approx(2.5f));
}

TEST_CASE("fill_const on float tensors", "[ops][fill]")
{
    TensorImpl t({2, 3}, ScalarType::Float32);

    SECTION("fractional value")
    {
        t.fill_const(2.5);
        const float* p = t.data_ptr<float>();
        for (size_t i = 0; i < t.total_size_; ++i)
            CHECK(p[i] == Catch::Approx(2.5f));
    }

    SECTION("negative value")
    {
        t.fill_const(-1.5);
        const float* p = t.data_ptr<float>();
        for (size_t i = 0; i < t.total_size_; ++i)
            CHECK(p[i] == Catch::Approx(-1.5f));
    }
}

TEST_CASE("fill_const on int tensors", "[ops][fill]")
{
    TensorImpl t({4}, ScalarType::Int32);   // use whatever your enum calls it
    t.fill_const(7);

    const int32_t* p = t.data_ptr<int32_t>();
    for (size_t i = 0; i < t.total_size_; ++i)
        CHECK(p[i] == 7);
}