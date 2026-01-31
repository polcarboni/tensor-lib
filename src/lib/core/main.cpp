#include "Tensor2.hpp"
#include <iostream>

int main()
{

    // ---------------------------- TYPES ----------------------------

    auto s1 = tensor::ScalarType::Float32;
    auto s2 = tensor::ScalarType::Int64;

    std::cout << "Scalar types:\n";
    std::cout << "  s1 = " << s1
              << ", element size = " << tensor::element_size(s1) << "\n";
    std::cout << "  s2 = " << s2
              << ", element size = " << tensor::element_size(s2) << "\n";

    tensor::Device cpu{tensor::DeviceType::CPU, 0};
    tensor::Device gpu{tensor::DeviceType::CUDA, 1};

    std::cout << "\nDevices:\n";
    std::cout << "  cpu = " << cpu << "\n";
    std::cout << "  gpu = " << gpu << "\n";

    // Explicit to_string usage
    std::string cpuStr = tensor::to_string(cpu);
    std::cout << "\nCPU as string: " << cpuStr << "\n";


    // ---------------------------- ALLOCATORS ----------------------------

    //Manual allocation with allocator
    try
    {
        tensor::CPUAllocator allocator;

        // allocate space for 16 doubles
        std::size_t num_elems = 16;
        std::size_t num_bytes = num_elems * sizeof(double);

        void* raw = allocator.allocate(num_bytes);
        std::cout << "Allocated " << num_bytes << " bytes at " << raw << "\n";

        // use the memory
        double* data = static_cast<double*>(raw);
        for (std::size_t i = 0; i < num_elems; ++i)
        {
            data[i] = static_cast<double>(i);
        }

        // verify contents
        for (std::size_t i = 0; i < num_elems; ++i)
        {
            std::cout << "data[" << i << "] = " << data[i] << "\n";
        }

        // free memory
        allocator.deallocate(raw);
        std::cout << "Memory deallocated\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }




    
    return 0;
}