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

    // ====================================== CORNER CASE TESTS ======================================

    // --- 1. SCALAR BROADCAST (shape {1} with multi-dim) ---
    std::cout << "\n[TEST] Scalar broadcast {1} + {3,3,3}:" << std::endl;
    auto scalar = tensor::TensorImpl({1}, 2.0);
    auto big    = tensor::TensorImpl({3,3,3}, 1.0);
    auto out_scalar_broadcast = tensor::add(scalar, big);
    std::cout << out_scalar_broadcast << std::endl; // expect all 3.0

    // NOT WORKING (only the first one is summed)

    // --- 2. SAME SHAPE (no broadcast needed) ---
    std::cout << "\n[TEST] Same shape {3,3} + {3,3}:" << std::endl;
    auto same_a = tensor::TensorImpl({3,3}, 1.5);
    auto same_b = tensor::TensorImpl({3,3}, 2.5);
    auto out_same = tensor::add(same_a, same_b);
    std::cout << out_same << std::endl; // expect all 4.0

    // WORKS

    // --- 3. IDENTITY (adding zero) ---
    std::cout << "\n[TEST] Identity: {3,3} + zeros {3,3}:" << std::endl;
    auto identity_a = tensor::TensorImpl({3,3}, 7.77);
    auto identity_b = tensor::TensorImpl({3,3}, 0.0);
    auto out_identity = tensor::add(identity_a, identity_b);
    std::cout << out_identity << std::endl; // expect all 7.77
    // WORKS

    // --- 4. NEGATIVE VALUES ---
    std::cout << "\n[TEST] Negative values {2,2} + {2,2}:" << std::endl;
    auto neg_a = tensor::TensorImpl({2,2}, -3.5);
    auto neg_b = tensor::TensorImpl({2,2},  3.5);
    auto out_neg = tensor::add(neg_a, neg_b);
    std::cout << out_neg << std::endl; // expect all 0.0
    // WORKS

    // --- 5. LARGE VALUE / OVERFLOW STRESS ---
    std::cout << "\n[TEST] Large values (overflow stress) {2,2}:" << std::endl;
    auto large_a = tensor::TensorImpl({2,2},  1e38f);
    auto large_b = tensor::TensorImpl({2,2},  1e38f);
    auto out_large = tensor::add(large_a, large_b);
    std::cout << out_large << std::endl; // expect inf (float overflow)
    // 199999993605713849301312521538346418176.0000

    // --- 6. VERY SMALL VALUES (underflow / precision) ---
    std::cout << "\n[TEST] Subnormal values {2,2}:" << std::endl;
    auto small_a = tensor::TensorImpl({2,2}, 1e-38f);
    auto small_b = tensor::TensorImpl({2,2}, 1e-38f);
    auto out_small = tensor::add(small_a, small_b);
    std::cout << out_small << std::endl; // expect ~2e-38
    // 0.0000

    // --- 7. 1D TENSORS ---
    std::cout << "\n[TEST] 1D tensors {5} + {5}:" << std::endl;
    auto vec_a = tensor::TensorImpl({5}, 1.0);
    auto vec_b = tensor::TensorImpl({5}, 2.0);
    auto out_vec = tensor::add(vec_a, vec_b);
    std::cout << out_vec << std::endl; // expect all 3.0
    // WORKS

    // --- 8. HIGH-RANK TENSORS ---
    std::cout << "\n[TEST] High-rank {2,2,2,2} + {1}:" << std::endl;
    auto rank4 = tensor::TensorImpl({2,2,2,2}, 1.0);
    auto rank0 = tensor::TensorImpl({1},       9.0);
    auto out_rank4 = tensor::add(rank4, rank0);
    std::cout << out_rank4 << std::endl; // expect all 10.0
    // WORKS

    // --- 9. MISMATCHED INCOMPATIBLE SHAPES (expect throw) ---
    std::cout << "\n[TEST] Incompatible shapes {2,3} + {3,2} (expect exception):" << std::endl;
    try {
        auto bad_a = tensor::TensorImpl({2,3}, 1.0);
        auto bad_b = tensor::TensorImpl({3,2}, 1.0);
        auto out_bad = tensor::add(bad_a, bad_b);
        std::cout << "ERROR: should have thrown!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Caught expected exception: " << e.what() << std::endl;
    }
    // WORKS

    // --- 10. BROADCAST ALONG BATCH DIM {4,1,3} + {1,2,3} ---
    std::cout << "\n[TEST] Broadcast {4,1,3} + {1,2,3}:" << std::endl;
    auto batch_a = tensor::TensorImpl({4,1,3}, 1.0);
    auto batch_b = tensor::TensorImpl({1,2,3}, 2.0);
    auto out_batch = tensor::add(batch_a, batch_b);
    std::cout << out_batch << std::endl; // expect shape {4,2,3}, all 3.0
    // WORKS

    // --- 11. SELF-ADDITION (aliasing) ---
    std::cout << "\n[TEST] Self-addition aliasing {3,3} + itself:" << std::endl;
    auto self_a = tensor::TensorImpl({3,3}, 5.0);
    auto out_self = tensor::add(self_a, self_a);
    std::cout << out_self << std::endl; // expect all 10.0
    // WORKS

    // --- 12. CUDA: MISMATCHED DEVICES (expect throw) ---
    std::cout << "\n[TEST] CUDA device mismatch (expect exception):" << std::endl;
    try {
        auto cpu_t  = tensor::TensorImpl({3,3}, 1.0);
        auto cuda_t = tensor::TensorImpl({3,3}, 1.0, tensor::ScalarType::Float32,
                                        {tensor::DeviceType::CUDA, 0});
        auto out_mixed = tensor::add(cpu_t, cuda_t);
        std::cout << "ERROR: should have thrown!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Caught expected exception: " << e.what() << std::endl;
    }
    // WORKS

    // --- 13. CUDA: SCALAR BROADCAST ---
    std::cout << "\n[TEST] CUDA scalar broadcast {1} + {3,3,3}:" << std::endl;
    auto cuda_scalar = tensor::TensorImpl({1},     2.0, tensor::ScalarType::Float32,
                                        {tensor::DeviceType::CUDA, 0});
    auto cuda_big    = tensor::TensorImpl({3,3,3}, 1.0, tensor::ScalarType::Float32,
                                        {tensor::DeviceType::CUDA, 0});
    auto out_cuda_scalar = tensor::add(cuda_scalar, cuda_big);
    std::cout << out_cuda_scalar << std::endl; // expect all 3.0
    

    // ====================================== IMPLICIT TYPE CONVERSION TESTS ======================================

    // --- 1. FLOAT32 + FLOAT64 (widening) ---
    std::cout << "\n[TEST] Float32 + Float64 widening {3,3} + {3,3}:" << std::endl;
    auto f32 = tensor::TensorImpl({3,3}, 1.1f, tensor::ScalarType::Float32);
    auto f64 = tensor::TensorImpl({3,3}, 2.2,  tensor::ScalarType::Float64);
    auto out_f32_f64 = tensor::add(f32, f64);
    std::cout << "result dtype (expect Float64): " << out_f32_f64.get_dtype() << std::endl;
    std::cout << out_f32_f64 << std::endl; // expect all ~3.3 in Float64

    // --- 4. BOOL + INT32 (bool promotes to int) ---
    std::cout << "\n[TEST] Bool + Int32 {2,2} + {2,2}:" << std::endl;
    auto bool_t = tensor::TensorImpl({2,2}, true,  tensor::ScalarType::Bool);
    auto int_t  = tensor::TensorImpl({2,2}, 4,     tensor::ScalarType::Int32);
    auto out_bool_int = tensor::add(bool_t, int_t);
    std::cout << "result dtype (expect Int32): " << out_bool_int.get_dtype() << std::endl;
    std::cout << out_bool_int << std::endl; // expect all 5

    // --- CUDA: FLOAT32 + FLOAT64 widening ---
    std::cout << "\n[TEST] CUDA Float32 + Float64 widening {3,3} + {3,3}:" << std::endl;
    auto cuda_f32 = tensor::TensorImpl({3,3}, 1.1f, tensor::ScalarType::Float32,
                                    {tensor::DeviceType::CUDA, 0});
    auto cuda_f64 = tensor::TensorImpl({3,3}, 2.2,  tensor::ScalarType::Float64,
                                    {tensor::DeviceType::CUDA, 0});
    auto out_cuda_f32_f64 = tensor::add(cuda_f32, cuda_f64);
    std::cout << "result dtype (expect Float64): " << out_cuda_f32_f64.get_dtype() << std::endl;
    std::cout << out_cuda_f32_f64 << std::endl; // expect all ~3.3 in Float64

    // --- CUDA: BOOL + INT32 (bool promotes to int) ---
    std::cout << "\n[TEST] CUDA Bool + Int32 {2,2} + {2,2}:" << std::endl;
    auto cuda_bool = tensor::TensorImpl({2,2}, true, tensor::ScalarType::Bool,
                                        {tensor::DeviceType::CUDA, 0});
    auto cuda_int  = tensor::TensorImpl({2,2}, 4,    tensor::ScalarType::Int32,
                                        {tensor::DeviceType::CUDA, 0});
    auto out_cuda_bool_int = tensor::add(cuda_bool, cuda_int);
    std::cout << "result dtype (expect Int32): " << out_cuda_bool_int.get_dtype() << std::endl;
    std::cout << out_cuda_bool_int << std::endl; // expect all 5


    return 0;
}

/**
 * REQUIRED TESTS
 * 
 * Different values (requires the definition of initializations).
 * Different types/type casting. 
 * 
 * 
 */