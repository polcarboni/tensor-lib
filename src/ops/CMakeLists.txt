# target_sources(tensor_lib PRIVATE
#                TensorIterator.cpp
#                Matmul.cpp)

add_library(tensor_ops STATIC
    TensorIterator.cpp
    Matmul.cpp
    TensorOps.cpp)

target_include_directories(tensor_ops PUBLIC
    ${CMAKE_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(tensor_ops PUBLIC
    tensor_core)