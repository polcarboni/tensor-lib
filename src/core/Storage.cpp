#include "core/Storage.hpp"
#include "core/Allocator.hpp"
#include <cstring>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace tensor {

    // -------------------- PRIVATE HELPERS -------------------- 

    Allocator* Storage::get_allocator(DeviceType type) {
        
        if (type == DeviceType::CPU) {
            static CPUAllocator cpu_instance;
            return &cpu_instance;
        } else if (type == DeviceType::CUDA) {
            static CUDAAllocator cuda_instance;
            return &cuda_instance;
        } else {
            throw std::runtime_error("Unsupported device or CUDA support not copmiled");
        }
    }


    void Storage::copy_data_from(const void* src) {
        
        if(!src || size_bytes_ == 0 || !data_) return;

        if (!data_) {
            data_ = get_allocator(device_.type)->allocate(size_bytes_);
        }

        DeviceType src_type = DeviceType::CPU;

        #ifdef USE_CUDA
            cudaPointerAttributes attr{};
            cudaError_t status = cudaPointerGetAttributes(&attr, src);

            if (status != cudaSuccess) {
                cudaGetLastError();
            } else {
                #if CUDART_VERSION >= 10000
                    if (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged)
                #else 
                    if (attr.memoryType == cudaMemoryTypeDevice)
                #endif
                        src_type = DeviceType::CUDA;
            }
            cudaGetLastError();
        #endif

        if (device_.type == DeviceType::CPU)
        {
            if(src_type == DeviceType::CPU) {
                std::memcpy(data_, src, size_bytes_);       /* CPU -> CPU */
            } else {
                #ifdef USE_CUDA
                    cudaMemcpy(data_, src, size_bytes_, cudaMemcpyDeviceToHost);    /* GPU -> CPU */
                #endif
            }
        } else if (device_.type == DeviceType::CUDA) {
            #ifdef USE_CUDA
                cudaMemcpyKind kind = (src_type == DeviceType::CUDA) ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
                cudaError_t err = cudaMemcpy(data_, src, size_bytes_, kind);    /* CPU/GPU -> GPU */
                if (err != cudaSuccess) {
                    throw std::runtime_error(cudaGetErrorString(err));
                }
            #endif
            
        }
    }

    // -------------------- CONSTRUCTORS / DESTRUCTOR -------------------- 

    // Empty constructor
    Storage::Storage(size_t size_bytes, Device device)
        : size_bytes_(size_bytes), device_(device), data_(nullptr) {
        
        if (size_bytes_ > 0) {
            data_ = get_allocator(device_.type)->allocate(size_bytes_);
        }
    }

    // Constructor from data pointer
    Storage::Storage(size_t size_bytes, Device device, const void* src)
        : size_bytes_(size_bytes), device_(device), data_(nullptr) {
        
        this->data_ = get_allocator(device_.type)->allocate(size_bytes_);
        this->copy_data_from(src);
    }

    // Desctructor
    Storage::~Storage() {
        if (data_) {
            get_allocator(device_.type)->deallocate(data_);
            data_ = nullptr;
        }
    }

    // Copy constructor
    Storage::Storage(const Storage& other)
        : size_bytes_(other.size_bytes_), device_(other.device_), data_(nullptr) {
            data_ = get_allocator(device_.type)->allocate(size_bytes_);
            copy_data_from(other.data_);
    }

    // Copy assignment operator
    Storage& Storage::operator=(const Storage& other) {
        if (this != &other) {
            if (data_) {
                get_allocator(device_.type)->deallocate(data_);
            }
            size_bytes_ = other.size_bytes_;
            device_ = other.device_;
            data_ = nullptr;
        }

        if (size_bytes_ > 0) {
            data_ = get_allocator(device_.type)->allocate(size_bytes_);
            copy_data_from(other.data_);
        }
        return *this;
    }

    // Move constructor
    Storage::Storage(Storage&& other) noexcept
        : data_(other.data_), size_bytes_(other.size_bytes_), device_(other.device_) {
        other.data_ = nullptr;
        other.size_bytes_ = 0;
    }

    // Move assignment operator
    Storage& Storage::operator=(Storage&& other) noexcept {
        if (this != &other) {
            if (data_) {
                get_allocator(device_.type)->deallocate(data_);
            }

            data_ = other.data_;
            size_bytes_ = other.size_bytes_;
            device_ = other.device_;
            
            other.data_ = nullptr;
            other.size_bytes_ = 0;
        }
        return *this;
    }

    Storage Storage::clone() const {
        return *this;
    }

    // -------------------- ACCESSORS -------------------- 

    void* Storage::data() const {
        return data_;
    }

    size_t Storage::nbytes() const {
        return size_bytes_;
    }

    Device Storage::device() const {
        return device_;
    }

} // namespace tensor
