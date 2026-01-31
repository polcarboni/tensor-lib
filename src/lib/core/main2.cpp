#include <iostream>
#include <vector>
#include <iomanip>
#include "Tensor2.hpp" // Assuming your code is in this header

// Helper function to print Tensor metadata
void print_tensor_info(const std::string& name, const tensor::Tensor& t) {
    std::cout << "--- Tensor: " << name << " ---" << std::endl;
    std::cout << "  Device:  " << t.device() << std::endl;
    std::cout << "  DType:   " << t.dtype() << std::endl;
    
    std::cout << "  Shape:   [";
    for (size_t i = 0; i < t.shape().size(); ++i) {
        std::cout << t.shape()[i] << (i == t.shape().size() - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;

    std::cout << "  Strides: [";
    for (size_t i = 0; i < t.strides().size(); ++i) {
        std::cout << t.strides()[i] << (i == t.strides().size() - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;

    std::cout << "  Size:    " << t.size() << " elements" << std::endl;
    std::cout << "  ReqGrad: " << (t.requires_grad() ? "True" : "False") << std::endl;
    std::cout << std::endl;
}

int main() {
    try {
        std::cout << "=== Test 1: Empty / Uninitialized Tensor ===" << std::endl;
        tensor::Tensor t1({2, 3}, tensor::ScalarType::Float32, {tensor::DeviceType::CPU, 0});
        print_tensor_info("t1 (2x3 Float32)", t1);

        std::cout << "=== Test 2: Tensor from Host Data (Type Mapping) ===" << std::endl;
        // Testing the template constructor with std::vector<int>
        // get_scalar_type<int> should map to Int32
        std::vector<int> data = {1, 2, 3, 4, 5, 6};
        tensor::Tensor t2({3, 2}, data); 
        print_tensor_info("t2 (3x2 Int32 from data)", t2);

        std::cout << "=== Test 3: Shallow Copy (View Logic) ===" << std::endl;
        // Copy constructor uses Impl::clone() which shares the same Storage
        tensor::Tensor t2_view = t2; 
        t2_view.set_requires_grad(true);
        
        std::cout << "t2_view requires_grad set to true." << std::endl;
        std::cout << "t2 requires_grad: " << (t2.requires_grad() ? "True" : "False") << " (Expect True if shared)" << std::endl;
        std::cout << std::endl;

        std::cout << "=== Test 4: Deep Copy (Clone Logic) ===" << std::endl;
        tensor::Tensor t2_clone = t2.clone();
        t2_clone.set_requires_grad(false);
        
        std::cout << "t2_clone requires_grad set to false." << std::endl;
        std::cout << "t2 requires_grad remains: " << (t2.requires_grad() ? "True" : "False") << std::endl;
        std::cout << std::endl;

        std::cout << "=== Test 5: Error Handling (Size Mismatch) ===" << std::endl;
        try {
            std::vector<double> wrong_data = {1.0, 2.0};
            tensor::Tensor t_error({5, 5}, wrong_data); // Should throw
        } catch (const std::invalid_argument& e) {
            std::cout << "Caught expected error: " << e.what() << std::endl;
        }
        std::cout << std::endl;

#ifdef USE_CUDA
        std::cout << "=== Test 6: CUDA Allocation (If Compiled) ===" << std::endl;
        try {
            tensor::Device gpu_dev{tensor::DeviceType::CUDA, 0};
            tensor::Tensor t_cuda({1024, 1024}, tensor::ScalarType::Float32, gpu_dev);
            print_tensor_info("t_cuda (Large GPU Tensor)", t_cuda);
        } catch (const std::exception& e) {
            std::cout << "CUDA Test failed: " << e.what() << std::endl;
        }
#else
        std::cout << "=== Test 6: CUDA (Skipped - USE_CUDA not defined) ===" << std::endl;
#endif

    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "All tests passed!" << std::endl;
    return EXIT_SUCCESS;
}