#include "TensorLib.hpp"
#include <iostream>

int main()
{
    auto t1 = tensor::ScalarType::Float32;
    auto t2 = tensor::ScalarType::Float64;
    
    std::cout << "Promoted type: " << promote_types(t1, t2) << std::endl;

    return 0;
}