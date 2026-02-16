add_subdirectory(core)
# add_subdirectory(ops)
add_subdirectory(autograd)

if (ENABLE_CUDA)
    # add_subdirectory(cuda)
endif()

add_library(tensor_lib INTERFACE)

target_link_libraries(tensor_lib INTERFACE
    tensor_core
    # tensor_ops
    tensor_autograd)

if (ENABLE_CUDA)
    # target_link_libraries(tensor_lib INTERFACE tensor_cuda)
endif()