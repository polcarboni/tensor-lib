#pragma once
#include <vector>
#include "Tensor.hpp"

namespace tensor
{
    template <typename T>
    class Module
    {
    protected:
        bool is_training = ture;
    public:
        virtual ~Module() = default;
        virtual Tensor<T> forward(const Tensor<T>& input) = 0;
        
        // Pointers to w and b. (for the optimizer) 
        virtual std::vector<Tensor<T>*> parameters() = 0;

        virtual void train() { is_training = true; }
        virtual void eval() { is_training = false; }

        // Forward operator overload
        Tensor<T> operator()(const Tensor<T>& input)
        {
            return forward(input);
        }
    };
}