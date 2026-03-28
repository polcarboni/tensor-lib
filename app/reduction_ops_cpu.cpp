/**
 * examples_reduction_cpu.cpp
 *
 * Runnable usage examples for reduction operations on CPU.
 * Covers sum, mul, max, min — standard usage and corner cases.
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
//  1. sum
//     • global reduce (no axes)
//     • reduce along axis 0  (rows collapsed)
//     • reduce along axis 1  (cols collapsed)
//     • keepdims = true
//     • reduce over multiple axes
// ─────────────────────────────────────────────────────────────────────────────

void reduction_sum()
{
    std::cout << "\n---------- sum ----------\n";

    // 3×4 tensor: 0, 1, 2, …, 11
    Tensor t({3, 4}, ScalarType::Float32, CPU);
    t.fill_arange(0.0, 1.0);
    print("input (3×4, arange 0→11)", t);

    // Global reduce — single scalar
    Tensor s_all = t.sum();
    print("sum()  [global, scalar 66]", s_all);

    std::cout << std::endl;
    
    // Reduce rows (axis 0) → shape {4}
    Tensor s0 = t.sum({0});
    print("sum({0})  [shape {4}]", s0);
    std::cout << std::endl;
    
    // Reduce cols (axis 1) → shape {3}
    Tensor s1 = t.sum({1});
    print("sum({1})  [shape {3}]", s1);
    std::cout << std::endl;
    
    // keepdims preserves the reduced dimension as size-1
    Tensor s0k = t.sum({0}, /*keepdims=*/true);
    print("sum({0}, keepdims)  [shape {1,4}]", s0k);
    std::cout << std::endl;
    
    Tensor s1k = t.sum({1}, /*keepdims=*/true);
    print("sum({1}, keepdims)  [shape {3,1}]", s1k);
    std::cout << std::endl;
    
    // Reduce over both axes → same as global but keepdims gives {1,1}
    Tensor s01  = t.sum({0, 1});
    print("sum({0,1})  [scalar 66]", s01);
    std::cout << std::endl;
    
    Tensor s01k = t.sum({0, 1}, /*keepdims=*/true);
    print("sum({0,1}, keepdims)  [shape {1,1}]", s01k);
    std::cout << std::endl;
}

// ─────────────────────────────────────────────────────────────────────────────
//  2. mul (product reduction)
//     • global reduce
//     • per-axis, keepdims variants
//     • corner case: tensor containing a zero  (product must be 0)
//     • corner case: 1-element tensor
// ─────────────────────────────────────────────────────────────────────────────

void reduction_mul()
{
    std::cout << "\n---------- mul (product) ----------\n";

    // 2×3 tensor: 1, 2, 3, 4, 5, 6  — product = 720
    Tensor t({2, 3}, ScalarType::Float32, CPU);
    t.fill_arange(1.0, 1.0);
    print("input (2×3, arange 1→6)", t);

    Tensor p_all = t.mul();
    print("mul()  [global, scalar 720]", p_all);

    Tensor p0 = t.mul({0});
    print("mul({0})  [shape {3}: 4,10,18]", p0);

    Tensor p1 = t.mul({1});
    print("mul({1})  [shape {2}: 6,120]", p1);

    Tensor p0k = t.mul({0}, /*keepdims=*/true);
    print("mul({0}, keepdims)  [shape {1,3}]", p0k);

    // Corner case: zero element — whole product collapses to 0
    std::cout << "\n  -- corner: tensor with a zero --\n";
    Tensor tz({2, 3}, ScalarType::Float32, CPU);
    tz.fill_arange(0.0, 1.0);   // 0, 1, 2, 3, 4, 5
    print("input (contains zero)", tz);
    print("mul()  [should be 0]", tz.mul());

    // Corner case: 1-element tensor — product == the single value
    std::cout << "\n  -- corner: 1-element tensor --\n";
    Tensor t1({1, 1}, ScalarType::Float32, CPU);
    t1.fill_arange(7.0, 1.0);   // single value: 7
    print("input (1×1, value=7)", t1);
    print("mul()  [should be 7]", t1.mul());
    print("mul({0}, keepdims)  [shape {1,1}]", t1.mul({0}, true));
}

// ─────────────────────────────────────────────────────────────────────────────
//  3. max
//     • global max
//     • per-axis, keepdims variants
//     • corner case: all-negative values
//     • corner case: single-element tensor
//     • corner case: tensor with duplicate maximum
// ─────────────────────────────────────────────────────────────────────────────

void reduction_max()
{
    std::cout << "\n---------- max ----------\n";

    // 3×4 tensor: 0 … 11
    Tensor t({3, 4}, ScalarType::Float32, CPU);
    t.fill_arange(0.0, 1.0);
    print("input (3x4, arange 0→11)", t);

    print("max()   [global, scalar 11]", t.max());

    Tensor mx0 = t.max({0});
    print("max({0})  [shape {4}: 8,9,10,11]", mx0);

    Tensor mx1 = t.max({1});
    print("max({1})  [shape {3}: 3,7,11]", mx1);

    Tensor mx0k = t.max({0}, /*keepdims=*/true);
    print("max({0}, keepdims)  [shape {1,4}]", mx0k);

    // Corner case: all-negative values — max must still be the least-negative
    std::cout << "\n  -- corner: all-negative tensor --\n";
    Tensor tn({2, 3}, ScalarType::Float32, CPU);
    tn.fill_arange(-6.0, 1.0);   // -6, -5, -4, -3, -2, -1
    print("input (all negative)", tn);
    print("max()  [should be -1]", tn.max());
    print("max({0})  [shape {3}: -3,-2,-1]", tn.max({0}));

    // Corner case: duplicate maximum value
    std::cout << "\n  -- corner: duplicate maximum --\n";
    Tensor td({2, 3}, ScalarType::Float32, CPU);
    td.fill_arange(0.0, 1.0);   // 0,1,2,3,4,5
    // Manually overwrite last element to create a tie at 5 — if the API
    // supports scalar fill on a slice you'd do that here; otherwise we just
    // rely on the existing data where max=5 appears once.
    print("input (max=5, unique)", td);
    print("max({1})  [shape {2}: 2,5]", td.max({1}));

    // Corner case: 1-element tensor
    std::cout << "\n  -- corner: 1-element tensor --\n";
    Tensor t1({1, 1}, ScalarType::Float32, CPU);
    t1.fill_arange(42.0, 1.0);
    print("input (1×1, value=42)", t1);
    print("max()  [should be 42]", t1.max());
    print("max({0}, keepdims)  [shape {1,1}]", t1.max({0}, true));
}

// ─────────────────────────────────────────────────────────────────────────────
//  4. min
//     • global min
//     • per-axis, keepdims variants
//     • corner case: all-positive values
//     • corner case: single-element tensor
//     • corner case: linspace with symmetric range (min should be negative end)
// ─────────────────────────────────────────────────────────────────────────────

void reduction_min()
{
    std::cout << "\n---------- min ----------\n";

    // 3×4 tensor: 0 … 11
    Tensor t({3, 4}, ScalarType::Float32, CPU);
    t.fill_arange(0.0, 1.0);
    print("input (3x4, arange 0→11)", t);

    print("min()   [global, scalar 0]", t.min());

    Tensor mn0 = t.min({0});
    print("min({0})  [shape {4}: 0,1,2,3]", mn0);

    Tensor mn1 = t.min({1});
    print("min({1})  [shape {3}: 0,4,8]", mn1);

    Tensor mn0k = t.min({0}, /*keepdims=*/true);
    print("min({0}, keepdims)  [shape {1,4}]", mn0k);

    Tensor mn1k = t.min({1}, /*keepdims=*/true);
    print("min({1}, keepdims)  [shape {3,1}]", mn1k);

    // Corner case: all-positive values — min must be the smallest positive
    std::cout << "\n  -- corner: all-positive tensor --\n";
    Tensor tp({2, 4}, ScalarType::Float32, CPU);
    tp.fill_linspace(1.0, 8.0);   // 1,2,3,4,5,6,7,8
    print("input (all positive)", tp);
    print("min()  [should be 1]", tp.min());
    print("min({1})  [shape {2}: 1,5]", tp.min({1}));

    // Corner case: symmetric range — min sits at the negative end
    std::cout << "\n  -- corner: symmetric linspace (-5→5) --\n";
    Tensor ts({2, 5}, ScalarType::Float32, CPU);
    ts.fill_linspace(-5.0, 5.0);
    print("input (linspace -5→5)", ts);
    print("min()  [should be -5]", ts.min());
    print("min({0})  [shape {5}]", ts.min({0}));

    // Corner case: 1-element tensor
    std::cout << "\n  -- corner: 1-element tensor --\n";
    Tensor t1({1, 1}, ScalarType::Float32, CPU);
    t1.fill_arange(3.0, 1.0);
    print("input (1x1, value=3)", t1);
    print("min()  [should be 3]", t1.min());
    print("min({0}, keepdims)  [shape {1,1}]", t1.min({0}, true));
}

// ─────────────────────────────────────────────────────────────────────────────
//  5. Cross-operation consistency checks
//     • sum == mul only for all-ones tensors
//     • max >= min always
//     • global reduce matches chained axis reduces
// ─────────────────────────────────────────────────────────────────────────────

void reduction_consistency()
{
    std::cout << "\n---------- consistency checks ----------\n";

    // All-ones: sum == N,  mul == 1
    Tensor ones({3, 4}, ScalarType::Float32, CPU);
    ones.fill_arange(1.0, 0.0);   // fill with constant 1 via step=0 if supported,
                                   // otherwise use fill_linspace(1,1)
    // Fallback: linspace(1→1) gives all ones for any count
    ones.fill_linspace(1.0, 1.0);
    print("all-ones (3x4)", ones);
    print("sum()  [should be 12]", ones.sum());
    print("mul()  [should be 1]",  ones.mul());
    print("max()  [should be 1]",  ones.max());
    print("min()  [should be 1]",  ones.min());

    // For any tensor: sum along axis 0 then sum along axis 0 again
    // (now axis 0 of the result) == global sum
    std::cout << "\n  -- chained axis reduce == global reduce --\n";
    Tensor t({3, 4}, ScalarType::Float32, CPU);
    t.fill_arange(0.0, 1.0);
    Tensor global_sum  = t.sum();
    Tensor chained_sum = t.sum({0}).sum({0});   // shape {4} → scalar
    print("global sum", global_sum);
    print("sum({0}) then sum({0})", chained_sum);

    // max of the per-row maxima == global max
    Tensor global_max  = t.max();
    Tensor chained_max = t.max({1}).max({0});
    print("global max", global_max);
    print("max({1}) then max({0})", chained_max);

    // min of the per-row minima == global min
    Tensor global_min  = t.min();
    Tensor chained_min = t.min({1}).min({0});
    print("global min", global_min);
    print("min({1}) then min({0})", chained_min);

    // keepdims shape test: reducing a {3,4} on axis 1 with keepdims → {3,1}
    // Summing that again on axis 0 with keepdims → {1,1} == global sum
    std::cout << "\n  -- keepdims chained --\n";
    Tensor step1 = t.sum({1}, /*keepdims=*/true);   // {3,1}
    std::cout << step1 << std::endl;
    Tensor step2 = step1.sum({0}, /*keepdims=*/true); // {1,1}
    print("sum({1},kd) then sum({0},kd)  [shape {1,1}, value 66]", step2);
}

// ─────────────────────────────────────────────────────────────────────────────
//  6. Higher-rank tensor (3-D)
//     Ensure reductions work beyond 2-D.
// ─────────────────────────────────────────────────────────────────────────────

void reduction_3d()
{
    std::cout << "\n---------- 3-D tensor reductions ----------\n";

    // Shape {2, 3, 4}  — 24 elements: 0 … 23
    Tensor t({2, 3, 4}, ScalarType::Float32, CPU);
    t.fill_arange(0.0, 1.0);
    print("input (2x3x4, arange 0→23)", t);
    std::cout << std::endl;

    // print("sum()      [scalar 276]",          t.sum());
    // std::cout << std::endl;
    // print("sum({0})   [shape {3,4}]",         t.sum({0}));
    // std::cout << std::endl;
    // print("sum({1})   [shape {2,4}]",         t.sum({1}));
    // std::cout << std::endl;
    print("sum({2})   [shape {2,3}]",         t.sum({2}));
    std::cout << std::endl;
    // print("sum({0,2}) [shape {3}]",           t.sum({0, 2}));
    // std::cout << std::endl;
    // print("sum({0,2}, keepdims) [{1,3,1}]",   t.sum({0, 2}, true));
    // std::cout << std::endl;
    
    print("max({2})   [shape {2,3}]",         t.max({2}));
    std::cout << std::endl;
    // print("min({0})   [shape {3,4}]",         t.min({0}));
    // std::cout << std::endl;
    print("mul({2})   [shape {2,3}]",         t.mul({2}));
    std::cout << std::endl;
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────

int main()
{
    // reduction_sum();
    // reduction_mul();
    // reduction_max();
    // reduction_min();
    // reduction_consistency();
    reduction_3d();
}