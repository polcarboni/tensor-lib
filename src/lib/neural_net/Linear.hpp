#pragma once
#include <random>
#include <cmath>
#include "Module.hpp"

template <typename T>
class Linear : public Module<T> {
private:
    Tensor<T> weight;
    Tensor<T> bias;

    
    void initialize_params(size_t in_features)
    {
        // TODO: parameter values initialization with He/Kaiming
        // (requires a helper function in the Tensor class)
    }

public:

    // TODO: this has only a simple constructor, no destructor, ...

    // TODO: Not sure what in_features and out features should be
    Linear(size_t in_features, size_t out_features)
        : weight({out_features, in_features})
        , bias({out_features})
        {
            initialize_params(in_features);
        }


    // FORWARD OPERATION
    Tensor<T> forward(const Tensor<T>& input) override
    {
        // output = input.matul(weight.transpose());
        // output.add(bias); broadcast addition

        // return output
    }


    std::vector<Tensor<T>*> parameters() override
    {
        return { &weight, &bias; }
    }
};