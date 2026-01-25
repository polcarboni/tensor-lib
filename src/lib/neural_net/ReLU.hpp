#pragma once
#include <algorithm>
#include "Module.hpp"

template <typename T>
class ReLU : public Module<T>
{
public:
    ReLU() = default;

    Tensor<T> forward(const Tensor<T>& input) override
    {
        // Copy shape/metadata(?) from the input Tensor
        
        // Application of reLU to all elements:
        // std::max(static_cast<T>(0), val);
    }

    std::vector<Tensor<T>*> parameters() override
    {
        return { };
    }
};