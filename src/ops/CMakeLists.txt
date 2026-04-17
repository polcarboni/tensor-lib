add_library(tensor_ops STATIC
    FillStorageOps.cpp
    BinaryOps.cpp
    ReductionOps.cpp
    UnaryOps.cpp
    MatMulOps.cpp)

target_include_directories(tensor_ops PUBLIC
    ${CMAKE_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR})

if (ENABLE_CUDA)
    add_subdirectory(cuda)
endif()