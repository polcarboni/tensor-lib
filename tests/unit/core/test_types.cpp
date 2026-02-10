#include <catch2/catch_test_macros.hpp>
#include "core/Types.hpp"
#include <sstream>

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
        CHECK(to_string(ScalarType::Float64) == "Float64");
        CHECK(to_string(ScalarType::Float32) == "Float32");
        CHECK(to_string(ScalarType::Int64) == "Int64");
        CHECK(to_string(ScalarType::Int32) == "Int32");
        CHECK(to_string(ScalarType::Bool) == "Bool");

        std::stringstream ss;
        ss << ScalarType::Bool;
        CHECK(ss.str() == "Bool");
    }

    SECTION("get_scalar_type")
    {
        CHECK(get_scalar_type<double>() == ScalarType::Float64);
        CHECK(get_scalar_type<float>() == ScalarType::Float32);
        CHECK(get_scalar_type<long long>() == ScalarType::Int64);
        CHECK(get_scalar_type<int64_t>() == ScalarType::Int64);
        CHECK(get_scalar_type<int32_t>() == ScalarType::Int32);
        CHECK(get_scalar_type<int>() == ScalarType::Int32);
        CHECK(get_scalar_type<bool>() == ScalarType::Bool);

        // CHECK(get_scalar_type<string>());
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

    SECTION("Device to_string and ostream operator")
    {
        
    }
}