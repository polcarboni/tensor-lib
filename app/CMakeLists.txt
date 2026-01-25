add_executable(${EXECUTABLE_NAME} main.cpp)

target_link_libraries(${EXECUTABLE_NAME} PRIVATE core_lib)

install(TARGETS ${EXECUTABLE_NAME})