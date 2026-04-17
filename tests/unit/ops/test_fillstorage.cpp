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
