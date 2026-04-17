add_executable(ops_unit_tests
    test_fillstorage.cpp)

target_link_libraries(ops_unit_tests
    PRIVATE
    Catch2::Catch2WithMain
    tensor_core
    tensor_ops)

catch_discover_tests(ops_unit_tests PROPERTIES
    LABELS "ops")