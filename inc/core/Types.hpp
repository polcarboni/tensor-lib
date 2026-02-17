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
    
    /* Runtime bytesize helper function */
    inline size_t element_size(ScalarType type)
    {
        switch(type) {
            case ScalarType::Float64: return sizeof(double);
            case ScalarType::Float32: return sizeof(float);
            case ScalarType::Int64:   return sizeof(int64_t);
            case ScalarType::Int32:   return sizeof(int32_t);
            case ScalarType::Bool:    return sizeof(bool);
            default: throw std::invalid_argument("Unsupported scalar type");
        }
    }

    inline constexpr ScalarType promote_types(ScalarType t1, ScalarType t2)
    {
        if (t1 == t2) return t1;
        if (t1 == ScalarType::Float64 || t2 == ScalarType::Float64) return ScalarType::Float64;
        if (t1 == ScalarType::Float32 || t2 == ScalarType::Float32) return ScalarType::Float32;
        if (t1 == ScalarType::Int64   || t2 == ScalarType::Int64)   return ScalarType::Int64;
        if (t1 == ScalarType::Int32   || t2 == ScalarType::Int32)   return ScalarType::Int32;
        return ScalarType::Bool;
    }

    inline std::string to_string(ScalarType type)
    {
        switch(type) {
            case ScalarType::Float64: return "Float64";
            case ScalarType::Float32: return "Float32";
            case ScalarType::Int64:   return "Int64";
            case ScalarType::Int32:   return "Int32";
            case ScalarType::Bool:    return "Bool";
            default: throw std::invalid_argument("Unsupported scalar type");
        }
    }

    inline std::ostream& operator<<(std::ostream& os, const ScalarType type)
    {
        return os << to_string(type);
    }

    /*
        auto t1 = get_scalar_type<double>();
    */
    template<typename T>
    inline constexpr ScalarType get_scalar_type()
    {
        if constexpr (std::is_same_v<T, double>)        return ScalarType::Float64;
        else if constexpr (std::is_same_v<T, float>)    return ScalarType::Float32;
        else if constexpr (std::is_same_v<T, int64_t>)  return ScalarType::Int64;
        else if constexpr (std::is_same_v<T, long long>) return ScalarType::Int64;
        else if constexpr (std::is_same_v<T, int32_t>)  return ScalarType::Int32;
        else if constexpr (std::is_same_v<T, int>)      return ScalarType::Int32;
        else if constexpr (std::is_same_v<T, bool>)     return ScalarType::Bool;
        else {
            static_assert(sizeof(T) == 0, "Unsupported C++ Type for Tensor ScalarType mapping");
        }
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
        switch(type) {
            case DeviceType::CPU:   return "CPU"; 
            case DeviceType::CUDA:  return "CUDA";
            default: throw std::invalid_argument("Unsupported device type");
        }
    }

    inline std::ostream& operator<<(std::ostream& os, DeviceType type)
    {
        return os << to_string(type);
    }

    // ------------------------------ DEVICE STRUCT  --------------------------------
    

    struct Device {
        DeviceType type = DeviceType::CPU;
        int index = 0;
    };

    inline bool operator==(const Device& lhs, const Device& rhs)
    {
        if(lhs.type == rhs.type && lhs.index == rhs.index ) {
            return true;
        } else {
            return false;
        }
    }  

    inline bool operator!=(const Device& lhs, const Device& rhs)
    {
        return !(lhs == rhs);
    }

    // TODO: oveload only with type (?)
    inline std::string to_string(const Device& device)
    {
        switch(device.type) {
            case DeviceType::CPU:   return "CPU"; 
            case DeviceType::CUDA:  return "CUDA";
            default: throw std::invalid_argument("Unsupported device type");
        }
    }
    
    inline std::ostream& operator<<(std::ostream& os, const Device& device)
    {
        return os << to_string(device.type);
    }

} // namespace tensor