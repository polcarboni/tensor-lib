/**
 * examples_binary_cpu.cpp
 *
 * Runnable usage examples for binary operations on CPU.
 * Mirrors the style of examples_unary_cpu.cpp.
 */

#include <iostream>
#include "TensorLib.hpp"

using namespace tensor;

static const Device CPU{DeviceType::CPU, 0};

// ─────────────────────────────────────────────────────────────────────────────
//  Helper — print a labelled tensor
// ─────────────────────────────────────────────────────────────────────────────

static void print(const std::string& label, const Tensor& t)
{
    std::cout << label << ":\n" << t << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  1. add / sub
// ─────────────────────────────────────────────────────────────────────────────

void binary_add_sub()
{
    std::cout << "\n---------- add & sub ----------\n";

    Tensor a({2, 3}, ScalarType::Float32, CPU);
    a.fill_arange(1.0, 1.0);    // 1, 2, 3, 4, 5, 6
    print("a", a);

    Tensor b({2, 3}, ScalarType::Float32, CPU);
    b.fill_arange(6.0, -1.0);   // 6, 5, 4, 3, 2, 1
    print("b", b);

    // Out-of-place
    Tensor sum = a.add(b);
    print("a.add(b)", sum);         // each element → 7

    Tensor diff = a.sub(b);
    print("a.sub(b)", diff);        // -5, -3, -1, 1, 3, 5

    // In-place — operate on independent clones so originals are preserved
    Tensor ca = a.clone();
    ca.add_inplace(b);
    print("add_inplace", ca);

    Tensor cs = a.clone();
    cs.sub_inplace(b);
    print("sub_inplace", cs);
}

// ─────────────────────────────────────────────────────────────────────────────
//  2. mul / div
// ─────────────────────────────────────────────────────────────────────────────

void binary_mul_div()
{
    std::cout << "\n---------- mul & div ----------\n";

    Tensor a({2, 4}, ScalarType::Float32, CPU);
    a.fill_linspace(1.0, 4.0);   // 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0 … (8 values)
    print("a (linspace 1→4)", a);

    Tensor b({2, 4}, ScalarType::Float32, CPU);
    b.fill_linspace(2.0, 2.0);   // constant 2.0 across all elements
    print("b (linspace 2→2)", b);

    // Out-of-place
    Tensor product = a.mul(b);
    print("a.mul(b)", product);     // each element doubled

    Tensor quotient = a.div(b);
    print("a.div(b)", quotient);    // each element halved

    // In-place
    Tensor cm = a.clone();
    cm.mul_inplace(b);
    print("mul_inplace", cm);

    Tensor cd = a.clone();
    cd.div_inplace(b);
    print("div_inplace", cd);
}

// ─────────────────────────────────────────────────────────────────────────────
//  3. exp (binary — element-wise power: a ^ b)
// ─────────────────────────────────────────────────────────────────────────────

void binary_exp()
{
    std::cout << "\n---------- exp (binary pow) ----------\n";

    // Bases: 1, 2, 3, 4  — kept small so results stay readable
    Tensor base({2, 2}, ScalarType::Float32, CPU);
    base.fill_arange(1.0, 1.0);   // 1, 2, 3, 4
    print("base", base);

    // Exponents: 0, 1, 2, 3
    Tensor exponent({2, 2}, ScalarType::Float32, CPU);
    exponent.fill_arange(0.0, 1.0);   // 0, 1, 2, 3
    print("exponent", exponent);

    // Out-of-place: 1^0=1, 2^1=2, 3^2=9, 4^3=64
    Tensor powered = base.exp(exponent);
    print("base.exp(exponent)", powered);

    // In-place
    Tensor cb = base.clone();
    cb.exp_inplace(exponent);
    print("exp_inplace", cb);
}

// ─────────────────────────────────────────────────────────────────────────────
//  4. Mixed signs — verifying correctness across sign boundaries
// ─────────────────────────────────────────────────────────────────────────────

void binary_mixed_signs()
{
    std::cout << "\n---------- mixed-sign operands ----------\n";

    // a spans negative → positive; b is its mirror
    Tensor a({2, 4}, ScalarType::Float32, CPU);
    a.fill_arange(-4.0, 1.0);   // -4, -3, -2, -1, 0, 1, 2, 3
    print("a", a);

    Tensor b({2, 4}, ScalarType::Float32, CPU);
    b.fill_arange(3.0, -1.0);   //  3,  2,  1,  0, -1, -2, -3, -4
    print("b", b);

    print("a.add(b)", a.add(b));   // all −1
    print("a.sub(b)", a.sub(b));   // -7, -5, -3, -1, 1, 3, 5, 7
    print("a.mul(b)", a.mul(b));   // element-wise products
}

// ─────────────────────────────────────────────────────────────────────────────
//  5. Chained / composed binary operations
//     Demonstrates that out-of-place binary ops return independent tensors
//     that can be fed directly into further operations.
// ─────────────────────────────────────────────────────────────────────────────

void binary_chained()
{
    std::cout << "\n---------- chained binary ops ----------\n";

    Tensor a({2, 3}, ScalarType::Float32, CPU);
    a.fill_linspace(1.0, 3.0);   // 1.0 … 3.0  (6 values)
    print("a (linspace 1→3)", a);

    Tensor b({2, 3}, ScalarType::Float32, CPU);
    b.fill_linspace(0.5, 1.5);   // 0.5 … 1.5  (6 values)
    print("b (linspace 0.5→1.5)", b);

    Tensor c({2, 3}, ScalarType::Float32, CPU);
    c.fill_linspace(2.0, 2.0);   // constant 2.0
    print("c (constant 2)", c);

    // (a + b) * c
    Tensor result1 = a.add(b).mul(c);
    print("(a + b) * c", result1);

    // (a * b) - (a / c)
    Tensor result2 = a.mul(b).sub(a.div(c));
    print("(a * b) - (a / c)", result2);

    // exp( a.sub(b) )  — combine binary sub with unary exp
    Tensor result3 = a.sub(b).exp();
    print("exp(a - b)", result3);
}

// ─────────────────────────────────────────────────────────────────────────────
//  6. In-place chaining — accumulate into a single tensor step-by-step
// ─────────────────────────────────────────────────────────────────────────────

void binary_inplace_accumulate()
{
    std::cout << "\n---------- in-place accumulation ----------\n";

    Tensor acc({3, 3}, ScalarType::Float32, CPU);
    acc.fill_arange(0.0, 1.0);   // 0 … 8
    print("initial acc", acc);

    Tensor ones({3, 3}, ScalarType::Float32, CPU);
    ones.fill_linspace(1.0, 1.0);   // all 1s
    print("ones", ones);

    Tensor scale({3, 3}, ScalarType::Float32, CPU);
    scale.fill_linspace(2.0, 2.0);   // all 2s
    print("scale (constant 2)", scale);

    // Step-by-step in-place: acc = (acc + 1) * 2
    acc.add_inplace(ones);
    print("after add_inplace(ones)", acc);   // 1 … 9

    acc.mul_inplace(scale);
    print("after mul_inplace(scale)", acc);  // 2 … 18

    acc.sub_inplace(ones);
    print("after sub_inplace(ones)", acc);   // 1 … 17

    acc.div_inplace(scale);
    print("after div_inplace(scale)", acc);  // 0.5 … 8.5
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────

int main()
{
    binary_add_sub();
    binary_mul_div();
    binary_exp();
    binary_mixed_signs();
    binary_chained();
    binary_inplace_accumulate();
}