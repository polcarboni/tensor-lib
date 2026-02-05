add_library(tensor_lib STATIC "")

# Allows the linker to mix C++ and CUDA
set_target_properties(tensor_lib PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden)

target_include_directories(tensor_lib PUBLIC ${CMAKE_SOURCE_DIR}/inc)

if (ENABLE_CUDA)
    set_target_properties(tensor_lib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

# add_subdirectory(lib)
add_subdirectory(lib2)