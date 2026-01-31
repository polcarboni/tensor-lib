// #pragma once
#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <stdexcept>
#include <cstdlib>  //aligned_alloc and free
#include <iostream>
#include <cstring>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

/*
    CURRENT TO-DO:
    Just implemented the type promotion, allocator should be checked. Further check in the storage class.
    STORAGE: to_string is not much useful but might be useful as part of a tensor print call. (Think about it backwards).

    ->: Start thinking about how to instance tensors. TARGET: Instance some tensors with different type and shapes on CPU (wait for GPU backend).
    <?>: Autograd meta unique_ptr vs shared_ptr (affects the constructors definition). is using shared_ptr a problem?
    <!>: USE an handle class Tensor and a TensorImpl class. PIMPL.

    NOT YET (but next): CUDA allocator, Operation dispatcher. (This seems quite complicated to do).
    type conversion have sense only with operations but I need to implement single tensors first.


    ------------------------------------------------------------------------------------------------

    Added the handle for PIMPL (correct but not tested). Added first version of CUDAAllocator.

    <?>: Something could be done for the device index (allocator factory that set device before allocation). 
*/


namespace tensor
{
    class Allocator;
    class Storage;
    class Tensor;
    struct AutogradMeta;
    class Node;
    class Edge;




    // -------------------------------------------------------------------------------------------------------------  
    //                                                 TYPES/HELPERS 
    // -------------------------------------------------------------------------------------------------------------  
    

    // ------------------------------------------------ SCALAR TYPE ------------------------------------------------ 

    enum class ScalarType { Float64, Float32, Int64, Int32, Bool };

    /* Return the higher type between two scalartypes*/
    inline constexpr ScalarType promote_types(ScalarType t1, ScalarType t2)
    {
        if (t1 == t2) return t1;
        if (t1 == ScalarType::Float64 || t2 == ScalarType::Float64) return ScalarType::Float64;
        if (t1 == ScalarType::Float32 || t2 == ScalarType::Float32) return ScalarType::Float32;
        if (t1 == ScalarType::Int64 || t2 == ScalarType::Int64)     return ScalarType::Int64;
        if (t1 == ScalarType::Int32 || t2 == ScalarType::Int32)     return ScalarType::Int32;
        return ScalarType::Bool;
    }


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

    template<typename T>
    inline constexpr ScalarType get_scalar_type() {
        if constexpr (std::is_same_v<T, double>) return ScalarType::Float64;
        else if constexpr (std::is_same_v<T, float>) return ScalarType::Float32;
        else if constexpr (std::is_same_v<T, int64_t>) return ScalarType::Int64;
        else if constexpr (std::is_same_v<T, long long>) return ScalarType::Int64;
        else if constexpr (std::is_same_v<T, int32_t>) return ScalarType::Int32;
        else if constexpr (std::is_same_v<T, int>) return ScalarType::Int32;
        else if constexpr (std::is_same_v<T, bool>) return ScalarType::Bool;
        else {
            static_assert(sizeof(T) == 0, "Unsupported C++ Type for Tensor ScalarType mapping");
        }
    }


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
    

    struct Device
    {
        DeviceType type;
        int index = 0;
    };

    inline std::string to_string(const Device& device)
    {
        return to_string(device.type) + "[" + std::to_string(device.index) + "]";
    }

    inline std::ostream& operator<<(std::ostream& os, const Device& device)
    {
        return os << to_string(device);
    }

    // ------------------------------ AUTOGRAD META STRUCT  --------------------------------
    
    struct AutogradMeta
    {
        std::shared_ptr<Tensor> grad_ = nullptr;    
        std::shared_ptr<Node> grad_fn_ = nullptr;
        std::weak_ptr<Node> grad_accumulator_;
        uint32_t version_ = 0;
    };




    // -------------------------------------------------------------------------------------------------------------  
    //                                                  ALLOCATORS 
    // -------------------------------------------------------------------------------------------------------------  


    /* Generic allocator interface: specializes for CPU and GPU */
    class Allocator
    {
    public:
        virtual void* allocate(size_t num_bytes) = 0;
        virtual void deallocate(void* ptr) = 0;
        virtual ~Allocator() = default;
    };

    /* CPU allocator: SIMD aligned */
    class CPUAllocator : public Allocator
    {
    public:
        static constexpr size_t ALIGNMENT = 64;  //AVX-512/SIMD
        void* allocate(size_t n) override
        {
            if (n == 0) return nullptr;
            size_t remainder = n % ALIGNMENT;
            size_t aligned_n = (remainder == 0) ? n : (n + ALIGNMENT - remainder);
            void *ptr = nullptr;

        #if defined(_MSC_VER) || defined(__MINGW32__)
            ptr = _aligned_malloc(aligned_n, ALIGNMENT);
        #else
            ptr = std::aligned_alloc(ALIGNMENT, aligned_n);
        #endif
            if (!ptr) throw std::bad_alloc();
            return ptr;
        }

        void deallocate(void* p) override
        {
        #if defined(_MSC_VER) || defined(__MINGW32__)
            _aligned_free(p);
        #else
            std::free(p);
        #endif
        }
    };

    /* CUDA GPU allocator */
    class CUDAAllocator : public Allocator {
    public:
        void* allocate(size_t n) override
        {
            if (n == 0) return nullptr;
            #ifdef USE_CUDA
                void *ptr = nullptr;
                cudaError_t err = cudaMalloc(&ptr, n);

                if (err != cudaSuccess) {
                    if (err == cudaErrorMemoryAllocation) {
                        throw std::bad_alloc();
                    }
                throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err));
                }

                return ptr;
            #else
                throw std::runtime_error("CUDAAllocator used but CUDA support is not copmiled");
                //TODO: default to CPU?
            #endif
        }

        void deallocate(void* p) override
        {
            if(!p) return;
            #ifdef USE_CUDA
                cudaError_t err = cudaFree(p);
                if (err != cudaSuccess) {
                    std::cerr << "CUDA Free failed: " << cudaGetErrorString(err) << std::endl;
                }
            #endif
        }
    };




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
        // std::shared_ptr<Allocator> allocator_; //TODO-fix: every single storage creates an allocator, change to static method
        
        static Allocator* get_allocator(DeviceType type) {
            if (type == DeviceType::CPU) {
                static CPUAllocator cpu_instance;
                return &cpu_instance;
            } else if (type == DeviceType::CUDA) {
                static CUDAAllocator cuda_instance;
                return &cuda_instance;
            }
            throw std::invalid_argument("Unknown device type");
        }

        void copy_data_from(const void * src) {
            if (!src || size_bytes_ == 0) return;

            if (device_.type == DeviceType::CPU) {
                std::memcpy(data_, src, size_bytes_);
            } else {
                #ifdef USE_CUDA
                    // Assumes src is host memory for the function
                    // Ensure correct device activation in multi-GPU setups 
                    cudaError_t err = cudaMemcpy(data_, src, size_bytes_, cudaMemcpyHostToDevice);
                    if (err != cudaSuccess) {
                        throw std::runtime_error(std::string("CUDA Memcpy HostToDevice failed: ") + cudaGetErrorString(err));
                    }
                #else
                    throw std::runtime_error("CUDA support not compiled.");
                #endif
            }
        }

    public:

        /* Uninitialized storage constructor (only allocates space) */
        Storage(size_t size_bytes, Device device)
            :  size_bytes_(size_bytes), device_(device)
        {
            if (size_bytes_ > 0) {
                data_ = get_allocator(device_.type)->allocate(size_bytes_);
            }
        }

        /* Constructor with initialization from host data */
        Storage(size_t size_bytes, Device device, const void* src)
            : Storage(size_bytes, device)
        {
            copy_data_from(src);
        }

        ~Storage()
        {
            if(data_) {
                get_allocator(device_.type)->deallocate(data_);
            }
        }


        // Copy constructor
        Storage(const Storage& other)
            : size_bytes_(other.size_bytes_), device_(other.device_)
        {
            if (size_bytes_ > 0 && other.data_)
            {
                data_ = get_allocator(device_.type)->allocate(size_bytes_);
                
                if(device_.type == DeviceType::CPU)
                {
                    std::memcpy(data_, other.data_, size_bytes_);
                }
                else if (device_.type == DeviceType::CUDA)
                {
                    #ifdef USE_CUDA
                        cudaError_t err = cudaMemcpy(data_, other.data_, size_bytes_, cudaMemcpyDeviceToDevice);
                        if(err != cudaSuccess) {
                            throw std::runtime_error(std::string("CUDA Memcpy DeviceToDevice failed: ") + cudaGetErrorString(err));
                        }
                    #endif
                }
            }
        }

        // Copy assignment
        Storage& operator=(const Storage& other)
        {
            if (this == &other) return *this;

            if (data_) {
                get_allocator(device_.type)->deallocate(data_);
            }

            size_bytes_ = other.size_bytes_;
            device_ = other.device_;

            if (size_bytes_ > 0 && other.data_) {
                data_ = get_allocator(device_.type)->allocate(size_bytes_);

                if (device_.type == DeviceType::CPU) {
                    std::memcpy(data_, other.data_, size_bytes_);
                } else if (device_.type == DeviceType::CUDA)
                {    
                    #ifdef USE_CUDA
                        cudaMemcpy(data_, other.data_, size_bytes_, cudaMemcpyDeviceToDevice);
                    #endif
                } else {
                    data_ = nullptr;
                }
            }
            return *this;
        }

        //  Move constructor
        Storage(Storage&& other) noexcept
            : data_(other.data_), size_bytes_(other.size_bytes_), device_(other.device_)
        {
            other.data_ = nullptr;
            other.size_bytes_ = 0;
        }

        Storage& operator=(Storage&& other) noexcept
        {
            if (this != &other) {
                if (data_) get_allocator(device_.type)->deallocate(data_);
                data_ = other.data_;
                size_bytes_ = other.size_bytes_;
                device_ = other.device_;
                other.data_ = nullptr;
                other.size_bytes_ = 0;
            }
            return *this;
        }

        Storage clone() const { return Storage(*this); }
  
        // -------------------- ACCESSORS -------------------- 

        void* data() const { return data_; }
        size_t nbytes() const { return size_bytes_; }
        Device device() const { return device_; }

        
    };


  

    // -------------------------------------------------------------------------------------------------------------  
    //                                                  TENSOR CLASS
    // ------------------------------------------------------------------------------------------------------------- 

    //TODO-fiX: methods should only be declared inside the class and defined after Tensor::impl (they require the full body to be defined)
    /*
        TENSOR CLASS
        */
    class Tensor
    {
    private:

        // ---------------------------- IMPLEMENTATION ---------------------------- 
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
        Tensor(std::unique_ptr<Impl>);
    
    public:
        
        // ---------------------------- DECLARATIONS ---------------------------- 

        Tensor();
        Tensor(const std::vector<size_t>&, ScalarType dtype, Device device);
        ~Tensor();
        Tensor(Tensor&&) noexcept = default;    // Move constructor
        Tensor& operator=(Tensor&&) noexcept = default;
        Tensor(const Tensor& other);    // Copy constructor
        Tensor clone() const;

        template<typename T>
        Tensor(const std::vector<size_t>&, const std::vector<T>&, Device device = {DeviceType::CPU, 0});

        // Accessors declarations
        const std::vector<size_t>& shape() const;
        const std::vector<size_t>& strides() const;
        ScalarType dtype() const;
        Device device() const;
        size_t size() const;
        size_t dims() const;
        bool requires_grad() const;
        void set_requires_grad(bool r) const;

    };




    // -------------------------------------------------------------------------------------------------------------  
    //                                        TENSOR IMPLEMENTATION CLASS
    // ------------------------------------------------------------------------------------------------------------- 

    // TODO: constructor to pass values to the storage (non-null initialization of vector)


    struct Tensor::Impl
    {
        std::shared_ptr<Storage> storage_ = nullptr;
        ScalarType dtype_;
        Device device_;
        size_t offset_ = 0;
        std::vector<size_t> shape_;
        std::vector<size_t> strides_;
        size_t total_size_ = 0;
        bool requires_grad_ = false;
        std::unique_ptr<AutogradMeta> autograd_meta_ = nullptr;

        void refresh_metadata()
        {
            if(shape_.empty()) {
                total_size_ = 0;
                return;
            }
            strides_.resize(shape_.size());
            size_t acc = 1;
            for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
                strides_[i] = acc;
                acc *= shape_[i];
            }
            total_size_ = acc;
        }
    
    
        /* Impl from pointer to values */
        Impl(const std::vector<size_t>& shape, ScalarType dtype, Device device, const void* src = nullptr)
            : dtype_(dtype), device_(device), shape_(shape)
        {
            refresh_metadata();
            storage_ = std::make_shared<Storage>(total_size_ * element_size(dtype), device, src);
        }

        //TODO: requires methods for deep copy and shallow copy (same storage)

        std::unique_ptr<Impl> clone() const
        {
            auto new_impl = std::make_unique<Impl>(shape_, dtype_, device_);
            new_impl->storage_ = storage_;
            new_impl->offset_ = offset_;
            new_impl->strides_ = strides_;
            new_impl->requires_grad_ = requires_grad_;
            
            return new_impl;
        }

    };

    // ------------------------------ TENSOR CLASS MEMBER METHODS ------------------------------ 

    // ---------------------------------- ACCESSORS ---------------------------------- 

    inline const std::vector<size_t>& Tensor::shape() const { return pimpl_->shape_; }
    inline const std::vector<size_t>& Tensor::strides() const { return pimpl_->strides_; }
    inline ScalarType Tensor::dtype() const { return pimpl_->dtype_; }
    inline Device Tensor::device() const { return pimpl_->device_; }
    inline size_t Tensor::size() const { return pimpl_->total_size_; }
    inline size_t Tensor::dims() const { return pimpl_->shape_.size(); }
    inline bool Tensor::requires_grad() const { return pimpl_->requires_grad_; }
    inline void Tensor::set_requires_grad(bool r) const { pimpl_->requires_grad_ = r; }


    // ---------------------------------- CONSTRUCTORS ---------------------------------- 
    
    // ---------------------------------- empty constructors ---------------------------------- 

    // Create tensor from existing implementation
    inline Tensor::Tensor(std::unique_ptr<Impl> impl)
        : pimpl_(std::move(impl)) {}

    // Empty tensor: no implementation
    inline Tensor::Tensor() : pimpl_(nullptr) { };   

    inline Tensor::Tensor(const std::vector<size_t>& shape, ScalarType dtype = ScalarType::Float32, Device device = {DeviceType::CPU, 0})
        : pimpl_(std::make_unique<Impl>(shape, dtype, device)) { };

    inline Tensor::~Tensor() = default;

    
    inline Tensor::Tensor(const Tensor& other) {        /* Shallow copy (view) */
        if (other.pimpl_) { pimpl_ = other.pimpl_->clone(); }
    }
    
    inline Tensor Tensor::clone() const {               /* Deep copy */
        //TODO-fix: constructor allocates memory
        auto new_tensor = Tensor(pimpl_->shape_, pimpl_->dtype_, pimpl_->device_);
        new_tensor.pimpl_->storage_ = std::make_shared<Storage>(pimpl_->storage_->clone());
        return new_tensor;
    }

    // ---------------------------------- overloaded (?) constructors ---------------------------------- 

    /* Construct tensor from: shape (std::vector<size_t>), values(std::vector<T>) and device */
    template<typename T>
    inline Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<T>& values, Device device)
    {
        size_t expected_elements = 1;
        for (auto s : shape) expected_elements *= s;

        if (values.size() != expected_elements) {
            //TODO-fix: more informative error message.
            throw std::invalid_argument("Values size does not match the shape");
        }

        ScalarType dtype = get_scalar_type<T>();

        pimpl_ = std::make_unique<Impl>(shape, dtype, device, values.data());
    }


    // -------------------------------------------------------------------------------------------------------------  
    //                                           OPERATION DISPATCHER CLASS
    // ------------------------------------------------------------------------------------------------------------- 

    class OpDispatcher
    {
    public:
        static Tensor add(const Tensor& lhs, const Tensor& rhs);
    };


    // -------------------------------------------------------------------------------------------------------------  
    //                                                  GRAD CLASSES ..... (to do later)
    // ------------------------------------------------------------------------------------------------------------- 




    class Edge
    {
    private:
        std::shared_ptr<Node> function_ = nullptr;
        uint32_t input_nr_ = 0;
    };


    class Node 
    {
    protected:
        std::vector<Edge> next_edges_;  //parents (?)
        std::string name_;
        size_t num_inputs_ = 0;
        //could track input metadata to ensure correct shapes
    };

} //namespace tensor