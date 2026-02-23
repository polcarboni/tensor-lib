add_library(tensor_core STATIC
    Allocator.cpp
    Storage.cpp
    TensorImpl.cpp
    TensorIterator.cpp
    Tensor.cpp)

target_include_directories(tensor_core PUBLIC
    ${CMAKE_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR})

if (ENABLE_CUDA)
    target_link_libraries(tensor_core PUBLIC
        PUBLIC
        CUDA::cudart)
endif()