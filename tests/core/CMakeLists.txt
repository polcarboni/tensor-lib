add_executable(unit_tests
    test_tensor.cpp
    test_operations.cpp)

target_link_libraries(unit_tests
    PRIVATE
    Catch2::Catch2WithMain
    tensor_lib)

catch_discover_tests(unit_tests PROPERTIES LABELS "unit")