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
    
    SECTION("element_size throws on EMPTY type")
    {
        CHECK_THROWS_AS(element_size(ScalarType::EMPTY), std::invalid_argument);
    }

    SECTION("promote_types logic")
    {
        // Same type
        CHECK(promote_types(ScalarType::Float64, ScalarType::Float64) == ScalarType::Float64);
        CHECK(promote_types(ScalarType::Float32, ScalarType::Float32) == ScalarType::Float32);
        CHECK(promote_types(ScalarType::Int64,   ScalarType::Int64)   == ScalarType::Int64);
        CHECK(promote_types(ScalarType::Int32,   ScalarType::Int32)   == ScalarType::Int32);
        CHECK(promote_types(ScalarType::Bool,    ScalarType::Bool)    == ScalarType::Bool);

        // Float64
        CHECK(promote_types(ScalarType::Float64, ScalarType::Float32) == ScalarType::Float64);
        CHECK(promote_types(ScalarType::Float64, ScalarType::Int64)   == ScalarType::Float64);
        CHECK(promote_types(ScalarType::Float64, ScalarType::Int32)   == ScalarType::Float64);
        CHECK(promote_types(ScalarType::Float64, ScalarType::Bool)    == ScalarType::Float64);
        
        CHECK(promote_types(ScalarType::Float32, ScalarType::Float64) == ScalarType::Float64);
        CHECK(promote_types(ScalarType::Int64,   ScalarType::Float64) == ScalarType::Float64);
        CHECK(promote_types(ScalarType::Int32,   ScalarType::Float64) == ScalarType::Float64);
        CHECK(promote_types(ScalarType::Bool,    ScalarType::Float64) == ScalarType::Float64);

        // Float32
        CHECK(promote_types(ScalarType::Float32, ScalarType::Int64)  == ScalarType::Float32);
        CHECK(promote_types(ScalarType::Float32, ScalarType::Int32)  == ScalarType::Float32);
        CHECK(promote_types(ScalarType::Float32, ScalarType::Bool)   == ScalarType::Float32);

        CHECK(promote_types(ScalarType::Int64,  ScalarType::Float32) == ScalarType::Float32);
        CHECK(promote_types(ScalarType::Int32,  ScalarType::Float32) == ScalarType::Float32);
        CHECK(promote_types(ScalarType::Bool,   ScalarType::Float32) == ScalarType::Float32);

        // Int64
        CHECK(promote_types(ScalarType::Int64, ScalarType::Int32) == ScalarType::Int64);
        CHECK(promote_types(ScalarType::Int64, ScalarType::Bool)  == ScalarType::Int64);

        CHECK(promote_types(ScalarType::Int32, ScalarType::Int64) == ScalarType::Int64);
        CHECK(promote_types(ScalarType::Bool,  ScalarType::Int64) == ScalarType::Int64);

        // Int32
        CHECK(promote_types(ScalarType::Int32, ScalarType::Bool) == ScalarType::Int32);
        CHECK(promote_types(ScalarType::Bool,  ScalarType::Int32) == ScalarType::Int32);
    }

    SECTION("to_string and ostream operator")
    {
        CHECK(to_string(ScalarType::Float64) == "Float64");
        CHECK(to_string(ScalarType::Float32) == "Float32");
        CHECK(to_string(ScalarType::Int64)   == "Int64");
        CHECK(to_string(ScalarType::Int32)   == "Int32");
        CHECK(to_string(ScalarType::Bool)    == "Bool");

        auto check_stream = [](ScalarType t, const std::string& expected) {
            std::stringstream ss;
            ss << t;
            CHECK(ss.str() == expected);
        };

        check_stream(ScalarType::Float64, "Float64");
        check_stream(ScalarType::Float32, "Float32");
        check_stream(ScalarType::Int64,   "Int64");
        check_stream(ScalarType::Int32,   "Int32");
        check_stream(ScalarType::Bool,    "Bool");

        CHECK_THROWS_AS(to_string(ScalarType::EMPTY), std::invalid_argument);
    }

    SECTION("get_scalar_type")
    {
        CHECK(get_scalar_type<double>()    == ScalarType::Float64);
        CHECK(get_scalar_type<float>()     == ScalarType::Float32);
        CHECK(get_scalar_type<long long>() == ScalarType::Int64);
        CHECK(get_scalar_type<int64_t>()   == ScalarType::Int64);
        CHECK(get_scalar_type<int32_t>()   == ScalarType::Int32);
        CHECK(get_scalar_type<int>()       == ScalarType::Int32);
        CHECK(get_scalar_type<bool>()      == ScalarType::Bool);

        // CHECK(get_scalar_type<string>()); compile-time static assert
    }
    
    SECTION("DISPATCH_ALL_TYPES correct branch execution")
    {
        auto dispatch_test = [](ScalarType t) {
            ScalarType result = ScalarType::EMPTY;
            DISPATCH_ALL_TYPES(t, "test", [&] {
                result = get_scalar_type<scalar_t>();
            });
            return result;
        };
        
        CHECK(dispatch_test(ScalarType::Float64) == ScalarType::Float64);
        CHECK(dispatch_test(ScalarType::Float32) == ScalarType::Float32);
        CHECK(dispatch_test(ScalarType::Int64)   == ScalarType::Int64);
        CHECK(dispatch_test(ScalarType::Int32)   == ScalarType::Int32);
        CHECK(dispatch_test(ScalarType::Bool)    == ScalarType::Bool);
    }
    
    SECTION("DISPATCH_ALL_TYPES throws on unsupported type")
    {
        CHECK_THROWS_AS(DISPATCH_ALL_TYPES(ScalarType::EMPTY, "test", [&]{}),
                                           std::runtime_error);
    }
}

TEST_CASE("Device and DeviceType", "[device]")
{
    SECTION("DeviceType to_string")
    {
        CHECK(to_string(DeviceType::CPU)  == "CPU");
        CHECK(to_string(DeviceType::CUDA) == "CUDA");
    }

    SECTION("DeviceType ostream operator")
    {
        std::stringstream ss;
        ss << DeviceType::CPU;
        CHECK(ss.str() == "CPU");
    
        ss.str("");
        ss << DeviceType::CUDA;
        CHECK(ss.str() == "CUDA");
    }

    SECTION("Device default construction")
    {
        Device d;
        CHECK(d.type  == DeviceType::CPU);
        CHECK(d.index == 0);
    }

    SECTION("Device equality and inequality")
    {
        Device cpu0{DeviceType::CPU,  0};
        Device cpu1{DeviceType::CPU,  1};
        Device cuda0{DeviceType::CUDA, 0};
        Device cuda1{DeviceType::CUDA, 1};

        // Equal to itself
        CHECK(cpu0  == cpu0);
        CHECK(cuda0 == cuda0);

        // Same type, same index
        CHECK(Device{DeviceType::CPU, 0} == cpu0);

        // Same type, different index
        CHECK(cpu0  != cpu1);
        CHECK(cuda0 != cuda1);

        // Different type, same index
        CHECK(cpu0  != cuda0);

        // Different type, different index
        CHECK(cpu1  != cuda0);
    }

    SECTION("Device to_string and ostream operator")
    {
        Device cpu{DeviceType::CPU, 0};
        Device cuda{DeviceType::CUDA, 0};

        CHECK(to_string(cpu)  == "CPU");
        CHECK(to_string(cuda) == "CUDA");

        std::stringstream ss;
        ss << cpu;
        CHECK(ss.str() == "CPU");

        ss.str("");
        ss << cuda;
        CHECK(ss.str() == "CUDA");
    }
}