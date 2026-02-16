#pragma once
#include "Types.hpp"

namespace tensor
{
    
    class Allocator;

    // -------------------------------------------------------------------------------------------------------------  
    //                                                  STORAGE CLASS
    // ------------------------------------------------------------------------------------------------------------- 

    /* 
        STORAGE CLASS 

        Does not actually contains the data but manages its allocation (constains a pointer to data)
        */
    class Storage
    {
    private:
        void* data_ = nullptr;
        size_t size_bytes_ = 0;
        Device device_;
        
        static Allocator* get_allocator(DeviceType type);
        void copy_data_from(const void * src);      /* to be called only after the allocation */

    public:
        
        Storage();
        ~Storage();
        Storage(size_t size_bytes, Device device);                      /* Uninitialized storage constructor (only allocates space) */
        Storage(size_t size_bytes, Device device, const void* src);     /* Constructor with initialization from host data */
        Storage(const Storage& other);                                  /* Copy constructor */

        Storage& operator=(const Storage& other);       /* Copy assignment */
        Storage(Storage&& other) noexcept;              /* Move constructor */
        Storage& operator=(Storage&& other) noexcept;

        Storage clone() const;
  
        // -------------------- ACCESSORS -------------------- 
        
        void* data() const;
        size_t nbytes() const;
        Device device() const;

        
        // -------------------- INITIALIZATION/FILLING -------------------- 
        
        // template <typename T>
        // void fill_from_value(T value);
        
        // void fill_from_buffer(void *src);
        
        // template <typename T>  // provide a function  for generating values
        // void generate();
    };
    
    // -------------------- VISUALIZATION -------------------- 
    inline std::string to_string(const Storage& storage)
    {
        return "Storage (size_bytes=" + std::to_string(storage.nbytes()) +
        ", device=" + to_string(storage.device()) + ")";
    }

    inline std::ostream& operator<<(std::ostream& os, const Storage& storage)
    {
        return os << to_string(storage);
    }

} // namespace tensor