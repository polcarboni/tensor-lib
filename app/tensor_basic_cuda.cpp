/**
 * examples_cuda.cpp
 *
 * Runnable usage examples for the Tensor library on CUDA.
 * Mirrors examples_cpu.cpp — same sections, same order.
 *
 */

#include <iostream>
#include "TensorLib.hpp"

using namespace tensor;

#ifdef USE_CUDA

static const Device CUDA0{DeviceType::CUDA, 0};

// ─────────────────────────────────────────────────────────────────────────────
//  1. Construction
// ─────────────────────────────────────────────────────────────────────────────

void construction()
{
    // Shape + device
    Tensor a({3, 4}, CUDA0);
    Tensor b({3, 4}, ScalarType::Float64, CUDA0);
    
    std::cout << a << std::endl;
    std::cout << std::endl;
    std::cout << b << std::endl;
    std::cout << std::endl;
    
    // Constant fill at construction time
    // Tensor zeros_like({3, 4}, CUDA0, 0.0);
    Tensor filled({3, 4}, ScalarType::Float32, CUDA0, 3.14);
    std::cout << filled << std::endl;
    std::cout << std::endl;
    
    // Factory helpers — pass CUDA device explicitly
    Tensor z = zeros({3, 4}, ScalarType::Float32, CUDA0);
    std::cout << z << std::endl;
    std::cout << std::endl;
    Tensor o = ones ({3, 4}, ScalarType::Float32, CUDA0);
    std::cout << o << std::endl;
    std::cout << std::endl;

    Tensor I = eye  (4, ScalarType::Float32, CUDA0);
    std::cout << I << std::endl;
    
    // Shallow copy (view) vs deep copy — both stay on device
    Tensor view  = b;          // shares device storage
    Tensor clone = b.clone();  // independent device copy
}

// ─────────────────────────────────────────────────────────────────────────────
//  2. Filling
// ─────────────────────────────────────────────────────────────────────────────

void filling()
{
    Tensor t({4, 4}, ScalarType::Float32, CUDA0);

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
    Tensor t({2, 3, 4}, ScalarType::Float32, CUDA0);
    t.fill_arange();

    std::cout << "\n---------- shape & layout ----------\n";
    std::cout << metadata_to_string(t) << "\n";

    std::vector<size_t> flat_shape{24};
    Tensor flat = t.view(flat_shape);
    std::cout << "view({24})      | " << metadata_to_string(flat) << "\n";

    std::vector<size_t> mat_shape{6, 4};
    Tensor mat = t.reshape(mat_shape);
    std::cout << "reshape({6,4})  | " << metadata_to_string(mat) << "\n";

    Tensor c = mat.contiguous();
    std::cout << "contiguous: " << c.is_contiguous() << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────

int main()
{
    construction();
    filling();
    shape_and_layout();
}

#else

int main()
{
    std::cout << "CUDA not available — skipping CUDA examples.\n";
}

#endif // TENSOR_HAS_CUDA