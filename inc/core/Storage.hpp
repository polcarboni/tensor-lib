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
        void copy_data_from(const void * src);      /* to be called only after the allocation */

    public:

        Storage(size_t size_bytes, Device device);                      /* Uninitialized storage constructor (only allocates space) */
        Storage(size_t size_bytes, Device device, const void* src);     /* Constructor with initialization from host data */
        Storage(const Storage& other);                                  /* Copy constructor */
        ~Storage();

        Storage& operator=(const Storage& other);       /* Copy assignment */
        Storage(Storage&& other) noexcept;              /* Move constructor */
        Storage& operator=(Storage&& other) noexcept;

        Storage clone() const;
  
        // -------------------- ACCESSORS -------------------- 

        void* data() const;
        size_t nbytes() const;
        Device device() const;
    };

} // namespace tensor