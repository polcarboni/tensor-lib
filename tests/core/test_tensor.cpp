#include <catch2/catch_test_macros.hpp>
#include "TensorLib.hpp"

// using namespace tensor_old;

// TEST_CASE("Tensor construction", "[tensor]")
// {
//     SECTION("Empty constructor")
//     {
//         Tensor<float> t;
//         REQUIRE(t.size() == 0);
//         REQUIRE(t.ndim() == 0);
//         REQUIRE(t.data() == nullptr);
//     }

//     SECTION("Shape and default value constructor") {
//         Tensor<int> t({2, 3}, 5);
//         REQUIRE(t.size() == 6);
//         REQUIRE(t.ndim() == 2);
//         REQUIRE(t.shape() == std::vector<size_t>{2, 3});
        
//         for(size_t i = 0; i < t.size(); ++i) {
//             REQUIRE(t[i] == 5);
//         }
//     }
// }


// TEST_CASE("Static Factories (Zeros and Ones)", "[tensor]")
// {
    
// }