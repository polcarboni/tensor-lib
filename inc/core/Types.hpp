#pragma once
#include <cstddef>
#include <iostream>

namespace tensor
{

    // -------------------------------------------------------------------------------------------------------------  
    //                                                 TYPES/HELPERS 
    // -------------------------------------------------------------------------------------------------------------  
    

    // ------------------------------------------------ SCALAR TYPE ------------------------------------------------ 

    enum class ScalarType { Float64, Float32, Int64, Int32, Bool };

    /* Return the higher type between two scalartypes*/
    inline constexpr ScalarType promote_types(ScalarType lhs, ScalarType rhs)
    {
        // dummy
        return lhs;
    }

    /* Runtime bytesize helper function */
    inline size_t element_size(ScalarType type)
    {

    }

    inline std::string to_string(ScalarType type)
    {

    }

    inline std::ostream& operator<<(std::ostream& os, const ScalarType type)
    {

    }

    // TODO: no param?
    template<typename T>
    inline constexpr ScalarType get_scalar_type()
    {
        auto x = ScalarType();
        return x;
    }

    //  DISPATCHER MACRO
    #define DISPATCH_ALL_TYPES(TYPE, NAME, ...) \
        [&] { \
            switch(TYPE) { \
                case ScalarType::Float32: {using scalar_t = float; return __VA_ARGS__(); break; } \
                case ScalarType::Float64: {using scalar_t = double; return __VA_ARGS__(); break; } \
                case ScalarType::Int32: {using scalar_t = int32_t; return __VA_ARGS__(); break; } \
                case ScalarType::Int64: {using scalar_t = int64_t; return __VA_ARGS__(); break; } \
                case ScalarType::Bool: {using scalar_t = bool; return __VA_ARGS__(); break; } \
                default: throw std::runtime_error(std::string(NAME) + " not implemented for " + to_string(TYPE)); \
                } \
            } ()


    // --------------------------------- DEVICE TYPE --------------------------------- 

    enum class DeviceType { CPU, CUDA };
    
    inline std::string to_string(DeviceType type)
    {

    }

    inline std::ostream& operator<<(std::ostream& os, DeviceType type)
    {

    }

    // ------------------------------ DEVICE STRUCT  --------------------------------
    

    struct Device {
        DeviceType type;
        int index = 0;
    };

    inline bool operator==(const Device& lhs, const Device& rhs)
    {

    }  

    inline bool operator!=(const Device& lhs, const Device& rhs)
    {

    }

    inline std::string to_string(const Device& device)
    {

    }
    
    inline std::ostream& operator<<(std::ostream& os, const Device& device)
    {
        
    }

} // namespace tensor