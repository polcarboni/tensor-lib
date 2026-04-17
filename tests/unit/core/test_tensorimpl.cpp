#pragma once
#include <catch2/catch_test_macros.hpp>
#include "core/TensorImpl.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace tensor;


using namespace tensor;

TEST_CASE("Tests for TensorImpl class", "[tensorimpl]")
{

    /**
     * TODO: some of these function could be refactored with explicit params use
     *       to have simple testing:
     * 
     * 
     */

    SECTION("refresh_metadata")
    {
        
    }

    SECTION("get_physical_offset")
    {

    }

    SECTION("Accessors testing")
    {

    }

    SECTION("to_string")
    {

    }

    SECTION("operator<< overload")
    {

    }
}