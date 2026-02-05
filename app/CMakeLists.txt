# add_executable(${EXECUTABLE_NAME_OLD} main_old.cpp)
# target_link_libraries(${EXECUTABLE_NAME_OLD} PRIVATE core_lib_old)

add_executable(${EXECUTABLE_NAME} main.cpp)
target_link_libraries(${EXECUTABLE_NAME} PRIVATE tensor_lib)

install(TARGETS ${EXECUTABLE_NAME})