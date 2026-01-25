add_library(core_lib INTERFACE)

target_include_directories(core_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_sources(core_lib INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/Autograd.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Operations.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Tensor.hpp")