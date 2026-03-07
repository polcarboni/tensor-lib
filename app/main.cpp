#include "TensorLib.hpp"
#include <iostream>
#include <thread>
#include <cuda_runtime.h>

#include <string>

int main()
{


    // =====================================================================================================
    //                                              Types Testing
    // =====================================================================================================
    
    auto t1 = tensor::ScalarType::Float32;
    auto t2 = tensor::ScalarType::Float64;
    auto t3 = tensor::ScalarType::Bool;
    
    std::cout << "Promoted type: " << promote_types(t1, t2) << std::endl;
    std::cout << "Promoted type: " << promote_types(t3, t1) << std::endl;
    std::cout << "Promoted type: " << promote_types(t3, t3) << std::endl;
    
    
    std::cout << "CPU logical cores: " << std::thread::hardware_concurrency() << std::endl;
    
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    for (size_t i = 0; i < deviceCount; ++i)
    {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "[" << i << "] " << prop.name << "(" << prop.major << prop.minor << ")" << std::endl;
    }
    

    // =====================================================================================================
    //                                            Storage Testing
    // =====================================================================================================
    
    auto a = tensor::Storage(64, {tensor::DeviceType::CPU, 0}) ;
    auto b = tensor::Storage(64, {tensor::DeviceType::CUDA, 0}) ;

    std::cout << "Storage A: " << a.nbytes() << " bytes, location: " << a.data() << std::endl;
    std::cout << "Storage B: " << b.nbytes() << " bytes, location: " << b.data() << std::endl;

    float data[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    auto cpu_storage = tensor::Storage(sizeof(data), {tensor::DeviceType::CPU, 0}, data);
    for (size_t i = 0; i < sizeof(data); i++)
        data[i] += 0.5f;
    auto cuda_storage = tensor::Storage(sizeof(data), {tensor::DeviceType::CUDA, 0}, data);

    // Check the allocated storage data.
    for (size_t i = 0; i < 8; ++i)
    {
        if (i==0) std::cout << "CPU data: [";
        std::cout << std::fixed << static_cast<float*>(cpu_storage.data())[i];
        (i < 7) ?  std::cout << ", " : std::cout << "]" <<std::endl;
    }

    float cuda_data[8] = {0.0f};
    cudaMemcpy(cuda_data, cuda_storage.data(), sizeof(data), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < 8; ++i)
    {
        if (i==0) std::cout << "CUDA data: [";
        std::cout << std::fixed << cuda_data[i];
        (i < 7) ?  std::cout << ", " : std::cout << "]" <<std::endl;
    }


    // =====================================================================================================
    //                                            TensorImpl
    // =====================================================================================================

    auto tensor_impl_a = tensor::TensorImpl();
    auto tensor_impl_b = tensor::TensorImpl({2,2}, 3.214f);
    
    std::cout << tensor_impl_a << std::endl;
    std::cout << tensor_impl_b << std::endl;

    auto impl_c = tensor::TensorImpl({3,3,3}, 4.321, tensor::ScalarType::Int32);
    auto impl_d = tensor::TensorImpl({3,3,3,3}, 4, tensor::ScalarType::Int32);

    std::cout << impl_c << std::endl;
    std::cout << impl_d << std::endl;

    // Tensor Impl on CUDA

    auto impl_cuda = tensor::TensorImpl({2,2}, 2.4, tensor::ScalarType::Float32, {tensor::DeviceType::CUDA, 0});
    std::cout << impl_cuda << std::endl;



    // ====================================== BINARY OPERATION CPU TEST ======================================
    
    std::cout << "\nADDING OPERATION: operators:" << std::endl;
    
    auto lhs = tensor::TensorImpl({3,3,3}, 4.321);
    auto rhs = tensor::TensorImpl({1,3}, 5.121);
    
    std::cout << lhs << std::endl;
    std::cout << "\n" << rhs << std::endl;
    
    std::cout << "\nresult:" << std::endl;
    
    auto out = tensor::add(lhs, rhs);
    std::cout << out << std::endl;


    // ====================================== BINARY OPERATION CUDA TEST ======================================
    
    std::cout << "\nADDING OPERATION CUDA: operators:" << std::endl;
    
    auto rhs_cuda = tensor::TensorImpl({1,3},   5.121, tensor::ScalarType::Float32, {tensor::DeviceType::CUDA, 0});
    auto lhs_cuda = tensor::TensorImpl({3,3,3}, 4.321, tensor::ScalarType::Float32, {tensor::DeviceType::CUDA, 0});
    
    std::cout << lhs_cuda << std::endl;
    std::cout << "\n" << rhs_cuda << std::endl;
    
    std::cout << "\nresult:" << std::endl;
    
    auto out_cuda = tensor::add(lhs_cuda, rhs_cuda);
    std::cout << out_cuda << std::endl;


    std::vector<size_t> red_axes = {};
    auto reduced   = tensor::sum(out, red_axes, false);
    auto reduced_2 = tensor::sum(out);

    return 0;
}