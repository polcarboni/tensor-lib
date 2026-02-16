add_executable(main main.cpp)

target_link_libraries(main PRIVATE tensor_lib)

if(ENABLE_CUDA)    
    target_link_libraries(main PRIVATE CUDA::cudart)
    set_target_properties(main PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native)
endif()

target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/inc)

install(TARGETS main)