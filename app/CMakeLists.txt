add_executable(main main.cpp)
target_link_libraries(main PRIVATE tensor_lib)
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/inc)
install(TARGETS main)

add_executable(tensor_basic_cpu tensor_basic_cpu.cpp)
target_link_libraries(tensor_basic_cpu PRIVATE tensor_lib)
target_include_directories(tensor_basic_cpu PRIVATE ${CMAKE_SOURCE_DIR}/inc)
install(TARGETS tensor_basic_cpu)

# ------------------------------ UNARY OPS ------------------------------

add_executable(unary_ops_cpu unary_ops_cpu.cpp)
target_link_libraries(unary_ops_cpu PRIVATE tensor_lib)
target_include_directories(unary_ops_cpu PRIVATE ${CMAKE_SOURCE_DIR}/inc)
install(TARGETS unary_ops_cpu)

# ------------------------------ BINARY OPS ------------------------------

add_executable(binary_ops_cpu binary_ops_cpu.cpp)
target_link_libraries(binary_ops_cpu PRIVATE tensor_lib)
target_include_directories(binary_ops_cpu PRIVATE ${CMAKE_SOURCE_DIR}/inc)
install(TARGETS binary_ops_cpu)

# ---------------------------- REDUCTION OPS -----------------------------

add_executable(reduction_ops_cpu reduction_ops_cpu.cpp)
target_link_libraries(reduction_ops_cpu PRIVATE tensor_lib)
target_include_directories(reduction_ops_cpu PRIVATE ${CMAKE_SOURCE_DIR}/inc)
install(TARGETS reduction_ops_cpu)

# ------------------------------ CUDA DEMOS ------------------------------

if(ENABLE_CUDA)
  foreach(demo tensor_basic_cuda unary_ops_cuda binary_ops_cuda)
    add_executable(${demo} ${demo}.cpp)
    target_link_libraries(${demo} PRIVATE tensor_lib)
    target_include_directories(${demo} PRIVATE ${CMAKE_SOURCE_DIR}/inc)
    install(TARGETS ${demo})
  endforeach()
endif()