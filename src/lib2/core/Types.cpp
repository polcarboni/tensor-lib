// #include "core/Types.hpp"

// namespace tensor {

//     // -------------------------------------------------------------------------------------------------------------  
//     //                                                TYPES/HELPERS 
//     // -------------------------------------------------------------------------------------------------------------  

//     // ------------------------------------------------ SCALAR TYPE ------------------------------------------------ 

//     // constexpr ScalarType promote_types(ScalarType a, ScalarType b) {
//     //     // empty stub
//     //     return a;
//     // }

//     size_t element_size(ScalarType type) {
//         // empty stub
//         return 0;
//     }

//     std::string to_string(ScalarType type) {
//         // empty stub
//         return {};
//     }

//     std::ostream& operator<<(std::ostream& os, const ScalarType type) {
//         // empty stub
//         return os;
//     }

//     template<typename T>
//     constexpr ScalarType get_scalar_type() {
//         // empty stub
//         return ScalarType::Float32;
//     }

//     // --------------------------------- DEVICE TYPE --------------------------------- 

//     std::string to_string(DeviceType type) {
//         // empty stub
//         return {};
//     }

//     std::ostream& operator<<(std::ostream& os, DeviceType type) {
//         // empty stub
//         return os;
//     }

//     // ------------------------------ DEVICE STRUCT  --------------------------------

//     bool operator==(const Device& lhs, const Device& rhs) {
//         // empty stub
//         return false;
//     }

//     bool operator!=(const Device& lhs, const Device& rhs) {
//         // empty stub
//         return !(lhs == rhs);
//     }

//     std::string to_string(const Device& device) {
//         // empty stub
//         return {};
//     }

//     std::ostream& operator<<(std::ostream& os, const Device& device) {
//         // empty stub
//         return os;
//     }

// } // namespace tensor
