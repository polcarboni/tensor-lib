#pragma once
#include "Types.hpp"

namespace tensor
{
    
    class Allocator;

    class Storage
    {
    private:
        void* data_ = nullptr;
        size_t size_bytes_ = 0;
        Device device_;
        
        static Allocator* get_allocator(DeviceType type);               /* Selects correct allocator based on the device */
        void copy_data_from(const void * src);                          /* Copy data from source, to be called only after get_allocator->allocate() is used */

    public:
        Storage();
        ~Storage();
        Storage(size_t size_bytes, Device device);                      /* Uninitialized storage constructor (only allocates space) */
        Storage(size_t size_bytes, Device device, const void* src);     /* Constructor with initialization from host data */
        Storage(const Storage& other);                                  /* Copy constructor */

        Storage& operator=(const Storage& other);       
        Storage(Storage&& other) noexcept;
        Storage& operator=(Storage&& other) noexcept;

        Storage clone() const;                                          /* Creates a new Storage from current */
  
        // ------------------------------------------- ACCESSORS --------------------------------------------
        
        void*  data() const;
        size_t nbytes() const;
        Device device() const;

    };
    

    // ------------------------------------------- VISUALIZATION UTILITIES ------------------------------------------- 

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