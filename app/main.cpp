#include "Tensor.hpp"
#include "Operations.hpp"

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
    
    Tensor<float> op1({3,3}, 5.0f);
    Tensor<float> op2({3,3}, 4.0f);

    Tensor<float> result = tensor::element_wise(op1, op2, std::plus<float>{});
    Tensor<float> result_2 = op1 + op2;
    Tensor<float> result_3 = op1 - op2;
    result.print();
    std::cout << "\n";
    result_3.print();



    // Simple network
    neural::Linear fc1{784,128};
    neural::Linear fc2{128,10};

    
    return 0;
};