#pragma once
#include "Types.hpp"
#include "Allocator.hpp"

namespace tensor
{
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
        void copy_data_from(const void * src);

    public:

        /* Uninitialized storage constructor (only allocates space) */
        Storage(size_t size_bytes, Device device);
        /* Constructor with initialization from host data */
        Storage(size_t size_bytes, Device device, const void* src);
        ~Storage();
        Storage(const Storage& other); // Copy constructor

        
        Storage& operator=(const Storage& other);   // Copy assignment
    
        //  Move constructor
        Storage(Storage&& other) noexcept;
        Storage& operator=(Storage&& other) noexcept;

        Storage clone() const;
  
        // -------------------- ACCESSORS -------------------- 

        void* data() const;
        size_t nbytes() const;
        Device device() const;
    };

} // namespace tensor