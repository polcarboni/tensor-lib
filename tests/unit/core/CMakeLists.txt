add_executable(core_unit_tests
    test_types.cpp
    test_tensor.cpp
    test_allocator.cpp)

target_link_libraries(core_unit_tests
    PRIVATE
    Catch2::Catch2WithMain
    tensor_core)

catch_discover_tests(core_unit_tests PROPERTIES
    LABELS "unit")