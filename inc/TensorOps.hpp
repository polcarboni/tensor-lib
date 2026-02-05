#pragma once 
#include "core/Tensor.hpp"
#include "ops/Functors.hpp"
#include "ops/OpsImpl.hpp"

namespace tensor
{
    // ------------------------------- BINARY OPERATIONS -------------------------------
    
    Tensor add(const Tensor& lhs, const Tensor& rhs);
    Tensor mul(const Tensor& lhs, const Tensor& rhs);
    
    // ------------------------------- UNARY OPERATIONS -------------------------------
    
    Tensor ReLU(const Tensor& lhs); 
    
    // ------------------------------- REDUCTION OPERATIONS -------------------------------
    
    Tensor sum(const Tensor& lhs, std::vector<size_t> dims, bool keepdim);
    Tensor mean(const Tensor& lhs, std::vector<size_t> dims, bool keepdim);
    
    
    // ------------------------------- OPERATORS OVERLOADING -------------------------------
    
    inline Tensor operator+(const Tensor& lhs, const Tensor& rhs)
    {
        return add(lhs,rhs);
    }

    inline Tensor operator*(const Tensor& lhs, const Tensor& rhs)
    {
        return mul(lhs, rhs);
    }
 
} // namespace tensor