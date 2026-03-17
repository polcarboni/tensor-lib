add_executable(main main.cpp)
target_link_libraries(main PRIVATE tensor_lib)
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/inc)
install(TARGETS main)

add_executable(tensor_basic_cpu tensor_basic_cpu.cpp)
target_link_libraries(tensor_basic_cpu PRIVATE tensor_lib)
target_include_directories(tensor_basic_cpu PRIVATE ${CMAKE_SOURCE_DIR}/inc)
install(TARGETS tensor_basic_cpu)

add_executable(tensor_basic_cuda tensor_basic_cuda.cpp)
target_link_libraries(tensor_basic_cuda PRIVATE tensor_lib)
target_include_directories(tensor_basic_cuda PRIVATE ${CMAKE_SOURCE_DIR}/inc)
install(TARGETS tensor_basic_cuda)