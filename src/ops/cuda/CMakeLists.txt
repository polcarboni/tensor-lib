add_library(tensor_cuda STATIC
    FillStorageOps.cu)

target_include_directories(tensor_cuda PUBLIC
    ${CMAKE_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(tensor_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

target_link_libraries(tensor_cuda PUBLIC
    CUDA::cudart)