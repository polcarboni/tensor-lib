# target_sources(tensor_lib PRIVATE
#                Types.cpp
#                Allocator.cpp
#                Storage.cpp
#                TensorImpl.cpp
#                Tensor.cpp)

add_library(tensor_core STATIC
    Allocator.cpp
    Storage.cpp
    TensorImpl.cpp
    TensorIterator.cpp
    Dispatchers.cpp
    Tensor.cpp)

target_include_directories(tensor_core PUBLIC
    ${CMAKE_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(tensor_core PUBLIC
    PUBLIC
    CUDA::cudart)

# set_target_properties(tensor_core PROPERTIES
#     POSITION_INDEPENDENT_CODE ON)