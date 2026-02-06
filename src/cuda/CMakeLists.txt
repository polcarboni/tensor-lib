add_library(tensor_cuda STATIC
    TensorCUDA.cu)

# Allows separable compilation of .cu files (device function can call functions defined in different files)
# Requires tensor_lib to be STATIC
set_target_properties(tensor_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(tensor_cuda PUBLIC
    ${CMAKE_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(tensor_cuda PUBLIC
    tensor_core
    tensor_ops)

# Optional: If your CUDA code needs specific libraries (like cuBLAS or curand)
# find_package(CUDAToolkit REQUIRED)
# target_link_libraries(CppAI_lib PUBLIC CUDA::curand CUDA::cublas)