#include <catch2/catch_test_macros.hpp>
#include "core/Types.hpp"

using namespace tensor;

TEST_CASE("ScalarType helper functions", "[scalar_type]")
{
    SECTION("element returns correct byte sizes")
    {
        CHECK(element_size(ScalarType::Float64) == sizeof(double));
        CHECK(element_size(ScalarType::Float32) == sizeof(float));
        CHECK(element_size(ScalarType::Int64)   == sizeof(int64_t));
        CHECK(element_size(ScalarType::Int32)   == sizeof(int32_t));
        CHECK(element_size(ScalarType::Bool)    == sizeof(bool));
    }

    SECTION("promote_types logic")
    {
        CHECK(promote_types(ScalarType::Float32, ScalarType::Float32) == ScalarType::Float32);
        // Add others
    }

    SECTION("to_string and ostream operator")
    {
        // CHECK(to_string(ScalarType::Float32) == "Float32");
        // // ...

        // std::stringstream ss;
        // ss << ScalarType::Bool;
        // CHECK(ss.str() == "Bool");
    }

    SECTION("Macro throws and invalid type")
    {

    }

}

TEST_CASE("Device and DeviceType", "[device]")
{
    SECTION("Device equality and inequality")
    {

    }

    SECTION("Device to_string")
    {
        
    }
}