add_subdirectory(core)
add_subdirectory(ops)
# add_subdirectory(autograd)

if (ENABLE_CUDA)
    add_subdirectory(cuda)
endif()

target_include_directories(tensor_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/ops)

if (ENABLE_CUDA)
    target_include_directories(tensor_lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
endif()