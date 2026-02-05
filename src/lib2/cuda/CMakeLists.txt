# Allows separable compilation of .cu files (device function can call functions defined in different files)
# Requires tensor_lib to be STATIC
set_target_properties(tensor_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_sources(tensor_lib
               PRIVATE
               TensorCUDA.cu)

# Optional: If your CUDA code needs specific libraries (like cuBLAS or curand)
# find_package(CUDAToolkit REQUIRED)
# target_link_libraries(CppAI_lib PUBLIC CUDA::curand CUDA::cublas)