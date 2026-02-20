#pragma once
#include <utility>

namespace tensor
{
    /* Forward declarations */
    
    enum class ScalarType;
    struct Device;
    struct TensorImpl;
    struct TensorIterator;

} // namespace tensor


namespace tensor::ops
{
    /**
     * The largest part of declarations is currently wrong as it does not requires device and dtype being passed
     * separately (with the single excpetion which is the nullary dispatcher).
     * 
     * The values will be extracted by the iterator from the provided operators.
     * TODO: check correct dispatchers type.
     * TODO: check which possible dispatchers are missing: matmul, backward versions
     * TODO: check correct dispatcher typing: inline must be void while the other ones have to return (forward one value, backward 2)
     * TODO: check what can be done with ternary operations. Backward? Technically possible (might be useful for GEMM operations).
     *       Ideally any type of dispatcher can be defined, it must be associated to operations that respect the TensorIterator building pattern
     *       used in the different operations.
     */


    // ---------------------------------------------- DISPATCHER IMPLEMENTATION ----------------------------------------------
    
    template <typename Op, typename... Args>
    void dispatch_impl_(TensorIterator& iter, Args&&... args);

    // TODO: check if required (probably not)
    // template <typename Op, typename... Args>
    // TensorImpl dispatch_nullary(Device device, ScalarType dtype, std::vector<size_t>& shape, Args... args);




    // ------------------------------------------------ UNARY DISPATCHERS ------------------------------------------------

    template <typename Op, typename... Args>
    TensorImpl dispatch_unary(TensorImpl& in, Args&&... args);

    // THIS ONE HAS THE CORRECT SIGNATURE
    template <typename Op, typename... Args>
    void dispatch_unary_inplace(TensorImpl& in, Args&&... args);

    template <typename Op, typename... Args>
    TensorImpl dispatch_unary_casting(TensorImpl& in, ScalarType dtype, Args&&... args);


    // ------------------------------------------------ BINARY DISPATCHERS ------------------------------------------------

    template <typename Op, typename... Args>
    TensorImpl dispatch_binary(TensorImpl& lhs, TensorImpl& rhs, Args&&... args);

    template <typename BackwardOp, typename... Args>
    std::pair<TensorImpl&, TensorImpl&> dispatch_binary_backward(TensorImpl& lhs, TensorImpl& rhs, Args&&... args);


    // ------------------------------------------------ TERNARY DISPATCHERS ------------------------------------------------

    template <typename Op, typename... Args>
    TensorImpl dispatch_ternary(TensorImpl& op_a, TensorImpl& op_b, TensorImpl& op_c, Args&&... args);


    // ------------------------------------------------ COMPARISON DISPATCHERS ------------------------------------------------

    template <typename Op, typename... Args>
    TensorImpl dispatch_comparison(TensorImpl& lhs, TensorImpl& rhs, Args&&... args);
    

    // ------------------------------------------------ REDUCTION DISPATCHERS ------------------------------------------------
    
    template <typename Op, typename... Args>
    TensorImpl dispatch_reduction(TensorImpl& tensor, Args&&... args);

    // TODO: inplace version required?

    // TODO: Matmul dispatcher required

} // namespace tensor::ops