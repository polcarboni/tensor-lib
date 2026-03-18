/**
 * examples_unary_cpu.cpp
 *
 * Runnable usage examples for unary operations on CPU.
 * Mirrors the style of examples_cpu.cpp.
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
//  1. neg / abs
// ─────────────────────────────────────────────────────────────────────────────

void unary_neg_abs()
{
    std::cout << "\n---------- neg & abs ----------\n";

    // Mix of positive and negative values: -3, -2, -1, 0, 1, 2
    Tensor t({2, 3}, ScalarType::Float32, CPU);
    t.fill_arange(-3.0, 1.0);   // -3, -2, -1, 0, 1, 2
    print("input", t);

    // Out-of-place
    Tensor n = t.neg();
    print("neg(t)", n);

    Tensor a = t.abs();
    print("abs(t)", a);

    // In-place — operate on independent clones so the original is preserved
    Tensor tn = t.clone();
    tn.neg_inplace();
    print("neg_inplace", tn);

    Tensor ta = t.clone();
    ta.abs_inplace();
    print("abs_inplace", ta);
}

// ─────────────────────────────────────────────────────────────────────────────
//  2. exp / log
// ─────────────────────────────────────────────────────────────────────────────

void unary_exp_log()
{
    std::cout << "\n---------- exp & log ----------\n";

    // Small positive values so log is well-defined on the same tensor
    Tensor t({2, 4}, ScalarType::Float32, CPU);
    t.fill_linspace(1.0, 2.0);   // 1.0 … 2.0 (8 values, all > 0)
    print("input (linspace 1→2)", t);

    // exp
    Tensor e = t.exp();
    print("exp(t)", e);

    Tensor te = t.clone();
    te.exp_inplace();
    print("exp_inplace", te);

    // log  (input is already > 0, so log is valid)
    Tensor l = t.log();
    print("log(t)", l);

    Tensor tl = t.clone();
    tl.log_inplace();
    print("log_inplace", tl);
}

// ─────────────────────────────────────────────────────────────────────────────
//  3. sigmoid
// ─────────────────────────────────────────────────────────────────────────────

void unary_sigmoid()
{
    std::cout << "\n---------- sigmoid ----------\n";

    // Span a range that exercises saturation on both sides
    Tensor t({2, 5}, ScalarType::Float32, CPU);
    t.fill_linspace(-4.0, 4.0);   // -4, -2, 0, 2, 4  (per row)
    print("input (linspace -4→4)", t);

    Tensor s = t.sigmoid();
    print("sigmoid(t)", s);

    Tensor ts = t.clone();
    ts.sigmoid_inplace();
    print("sigmoid_inplace", ts);
}

// ─────────────────────────────────────────────────────────────────────────────
//  4. tanh
// ─────────────────────────────────────────────────────────────────────────────

void unary_tanh()
{
    std::cout << "\n---------- tanh ----------\n";

    Tensor t({2, 5}, ScalarType::Float32, CPU);
    t.fill_linspace(-2.0, 2.0);
    print("input (linspace -2→2)", t);

    Tensor h = t.tanh();
    print("tanh(t)", h);

    Tensor th = t.clone();
    th.tanh_inplace();
    print("tanh_inplace", th);
}

// ─────────────────────────────────────────────────────────────────────────────
//  5. relu
// ─────────────────────────────────────────────────────────────────────────────

void unary_relu()
{
    std::cout << "\n---------- relu ----------\n";

    // Symmetric range so we can see the zero-clamp clearly
    Tensor t({3, 4}, ScalarType::Float32, CPU);
    t.fill_arange(-6.0, 1.0);   // -6, -5, …, 5
    print("input", t);

    Tensor r = t.relu();
    print("relu(t)", r);

    Tensor tr = t.clone();
    tr.relu_inplace();
    print("relu_inplace", tr);
}

// ─────────────────────────────────────────────────────────────────────────────
//  6. Chained / composed operations
//     Demonstrates that out-of-place ops return independent tensors that can
//     be fed into further unary (or later, binary) operations.
// ─────────────────────────────────────────────────────────────────────────────

void unary_chained()
{
    std::cout << "\n---------- chained unary ops ----------\n";

    // sigmoid( relu( t ) )  — only the positive half feeds into sigmoid
    Tensor t({2, 6}, ScalarType::Float32, CPU);
    t.fill_linspace(-3.0, 3.0);
    print("input (linspace -3→3)", t);

    Tensor chain = t.relu().sigmoid();
    print("sigmoid(relu(t))", chain);

    // exp( neg( abs( t ) ) )  — soft-decay envelope, always in (0, 1]
    Tensor decay = t.abs().neg().exp();
    print("exp(neg(abs(t)))", decay);
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────

int main()
{
    unary_neg_abs();
    unary_exp_log();
    unary_sigmoid();
    unary_tanh();
    unary_relu();
    unary_chained();
}