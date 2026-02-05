#include "Tensor.hpp"
#include "Operations.hpp"

// #include "Linear.hpp"
// #include "Module.hpp"
using namespace tensor_old; 

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

    Tensor<float> result = tensor_old::element_wise(op1, op2, std::plus<float>{});
    Tensor<float> result_2 = op1 + op2;
    Tensor<float> result_3 = op1 - op2;
    // result.print();
    // std::cout << "\n";
    // result_3.print();

    std::cout << result_2 << std::endl;
    std::cout << result_3 << std::endl;

    // Simple network
    // tensor_old::Linear<float> fc1(784,128);
    // tensor_old::Linear<float> fc2(128,10);


    // ----------------------- Neuron example -----------------------
    std::cout << "---- Neuron example ----" << std::endl;
    tensor_old::Tensor<float> input({1,10}, 1.0f);
    tensor_old::Tensor<float> weights({10,1}, 0.5f);
    tensor_old::Tensor<float> bias({1,1}, -11.0f);

    std::cout << "Input values:\n" << input << std::endl;
    std::cout << "Weights:\n" << weights << std::endl;
    std::cout << "Bias:\n" << bias << std::endl;

    // Matrix multiplication computation
    Tensor<float> weighted_sum = tensor_old::matmul_2D(input, weights);
    Tensor<float> z = weighted_sum + bias;

    std::cout << "Weighted sum:\n" << weighted_sum << std::endl;
    std::cout << "Weighted sum + bias:\n" << z << std::endl;


    // Activation ReLU function application
    Tensor<float> output = tensor_old::ReLU(z);
    
    std::cout << "Output:\n" << output << std::endl;
    return 0;
};