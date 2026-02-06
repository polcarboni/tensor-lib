# target_sources(tensor_lib PRIVATE
#                Node.cpp
#                Edge.cpp
#                Autogradengine.cpp)

add_library(tensor_autograd STATIC
               Node.cpp
               Edge.cpp
               AutogradEngine.cpp)

target_include_directories(tensor_autograd PUBLIC
    ${CMAKE_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(tensor_autograd PUBLIC
    tensor_core
    tensor_ops)