add_executable(main main.cpp)
target_link_libraries(main PRIVATE tensor_lib)
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/inc)

install(TARGETS main)