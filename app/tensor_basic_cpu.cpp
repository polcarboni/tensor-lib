/**
 * examples_cpu.cpp
 *
 * Runnable usage examples for the Tensor library on CPU.
 * Mirrors examples_cuda.cpp — same sections, same order.
 */

#include <iostream>
#include "TensorLib.hpp"

using namespace tensor;

static const Device CPU{DeviceType::CPU, 0};

// ─────────────────────────────────────────────────────────────────────────────
//  1. Construction
// ─────────────────────────────────────────────────────────────────────────────

void construction()
{
    // Shape only — dtype defaults to Float32
    Tensor a({3, 4});
    Tensor b({3, 4}, ScalarType::Float64);
    Tensor c({3, 4}, CPU);
    Tensor d({3, 4}, ScalarType::Float64, CPU);

    // Constant fill at construction time
    Tensor zeros_like({3, 4}, 0.0);
    Tensor ones_like ({3, 4}, 1.0);
    Tensor filled    ({3, 4}, ScalarType::Float32, CPU, 3.14);

    // Factory helpers
    Tensor z = zeros({3, 4});
    Tensor o = ones ({3, 4});
    Tensor I = eye  (4);

    // Shallow copy (view) vs deep copy
    Tensor view  = d;          // shares storage
    Tensor clone = d.clone();  // independent copy

    std::cout << "---------- construction ----------\n";
    std::cout << I << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  2. Filling
// ─────────────────────────────────────────────────────────────────────────────

void filling()
{
    Tensor t({4, 4}, ScalarType::Float32, CPU);

    std::cout << "\n---------- filling ----------\n";

    t.fill_const(7.0);
    std::cout << "fill_const(7):\n" << t << "\n";

    t.fill_arange(0.0, 1.0);          // 0, 1, 2, …  (start, step)
    std::cout << "fill_arange(0, step=1):\n" << t << "\n";

    t.fill_linspace(0.0, 1.0);        // evenly spaced from 0 to 1
    std::cout << "fill_linspace(0, 1):\n" << t << "\n";

    t.fill_rand(0.0, 1.0, 42);        // uniform in [low, high)
    std::cout << "fill_rand(0, 1):\n" << t << "\n";

    t.fill_rand_normal(0.0, 1.0, 42); // normal(mean, stddev)
    std::cout << "fill_rand_normal(mean=0, std=1):\n" << t << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  3. Shape & layout
// ─────────────────────────────────────────────────────────────────────────────

void shape_and_layout()
{
    Tensor t({2, 3, 4}, ScalarType::Float32, CPU);
    t.fill_arange();

    std::cout << "\n---------- shape & layout ----------\n";
    std::cout << t << "\n";

    // view — zero-copy reshape (only valid when contiguous)
    std::vector<size_t> flat_shape{24};
    Tensor flat = t.view(flat_shape);
    std::cout << flat << "\n";

    // FIX REQUIRED: Throws when reshaping for the different size of the
    // shape and strides 

    // reshape — copies if not contiguous, otherwise behaves like view
    std::vector<size_t> mat_shape{6, 4};
    Tensor mat = t.reshape(mat_shape);
    std::cout << "reshape({6,4})  | " << metadata_to_string(mat) << "\n";
    std::cout << mat << "\n";

    // contiguous() — returns a contiguous copy if needed
    Tensor cont = mat.contiguous();
    std::cout << "contiguous: " << cont.is_contiguous() << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  4. Indexing
// ─────────────────────────────────────────────────────────────────────────────

void indexing()
{
    Tensor t({3, 4}, ScalarType::Float32, CPU);
    t.fill_arange(0.0, 1.0);

    std::cout << "\n---------- indexing ----------\n";
    std::cout << "initial:\n" << t << "\n";

    // Multi-dim access via std::vector<size_t>
    float v00 = t.operator()<float>(0, 0);  // 0
    float v12 = t.operator()<float>(1, 2);  // 6

    // Multi-dim access via initializer_list
    float v23 = t.operator()<float>(2, 3);  // 11

    // Variadic (row, col, ...)
    float v11 = t.operator()<float>(1, 1);    // 5

    std::cout << "t(0,0)=" << v00 << "  t(1,2)=" << v12
              << "  t(2,3)=" << v23 << "  t(1,1)=" << v11 << "\n";

    // Write through any of the above, then print the result
    t.operator()<float>(0, 0) = 99.0f;
    t.operator()<float>(2, 3)   = -1.0f;
    std::cout << "after writes:\n" << t << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────

int main()
{
    construction();
    filling();
    shape_and_layout();
    indexing();
}