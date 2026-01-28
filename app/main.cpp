#include "Tensor.hpp"
#include "Operations.hpp"

#include "Linear.hpp"
#include "Module.hpp"
using namespace tensor; 

int main()
{

    Tensor<int> empty;
    empty.print();

    Tensor<int> a({2,2}, 5);
    a.print();

    try {
        Tensor<int> c({3,2}, {1,2,3});
    } catch (...) {
        std::cout << "Error catched" << std::endl;
    }

    //Operations testing
    
    Tensor<float> op1({3,3}, 5.0f);   //op1.activate_grad()
    Tensor<float> op2({3,3}, 4.0f);

    Tensor<float> result = tensor::element_wise(op1, op2, std::plus<float>{});
    Tensor<float> result_2 = op1 + op2;
    Tensor<float> result_3 = op1 - op2;
    result.print();
    std::cout << "\n";
    result_3.print();



    // Simple network
    tensor::Linear<float> fc1(784,128);
    tensor::Linear<float> fc2(128,10);


    // ----------------------- Neuron example -----------------------
    tensor::Tensor<float> input({1,10}, 1.0f);
    tensor::Tensor<float> weights({10,1}, 0.5f);
    tensor::Tensor<float> bias({1,1}, 0.1f);

    // Matrix multiplication computation
    Tensor<float> weighted_sum = tensor::matmul_2D(input, weights);
    Tensor<float> z = weighted_sum + bias;

    // Activation RELU function application
    Tensor<float> output = tensor::ReLU(z);
    
    return 0;
};