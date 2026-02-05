
# ------------------ INTEGRATION TESTS ------------------ 

add_executable(integration_tests
    test_pipeline.cpp)

target_link_libraries(integration_tests
    PRIVATE
    Catch2::Catch2WithMain
    tensor_lib)

catch_discover_tests(integration_tests PROPERTIES LABELS "integration")