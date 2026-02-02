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

namespace tensor
{
    class Allocator;
    class Storage;
    struct TensorImpl;
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
    inline constexpr ScalarType promote_types(ScalarType, ScalarType);

    /* Runtime bytesize helper function */
    inline size_t element_size(ScalarType type);

    inline std::string to_string(ScalarType type);

    inline std::ostream& operator<<(std::ostream& os, const ScalarType type);

    template<typename T>
    inline constexpr ScalarType get_scalar_type();

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
    
    inline std::string to_string(DeviceType type);
    inline std::ostream& operator<<(std::ostream& os, DeviceType type);

    // ------------------------------ DEVICE STRUCT  --------------------------------
    

    struct Device {
        DeviceType type;
        int index = 0;
    };

    inline bool operator==(const Device& lhs, const Device& rhs);   
    inline bool operator!=(const Device& lhs, const Device& rhs);

    inline std::string to_string(const Device& device);
    inline std::ostream& operator<<(std::ostream& os, const Device& device);


    // ------------------------------ AUTOGRAD META STRUCT  --------------------------------
    
    struct AutogradMeta {
        std::shared_ptr<Tensor> grad_ = nullptr;    
        std::shared_ptr<Node> grad_fn_ = nullptr;
        std::weak_ptr<Node> grad_accumulator_;
        uint32_t version_ = 0;
    };

    // -------------------------------------------------------------------------------------------------------------  
    //                                                  ALLOCATORS 
    // -------------------------------------------------------------------------------------------------------------  


    /* Generic allocator interface: specializes for CPU and GPU */
    class Allocator {
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
        void* allocate(size_t n) override;
        void deallocate(void* p) override;
    };

    /* CUDA GPU allocator */
    class CUDAAllocator : public Allocator {
    public:
        void* allocate(size_t n) override;
        void deallocate(void* p) override;
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


  

    // -------------------------------------------------------------------------------------------------------------  
    //                                        TENSOR IMPLEMENTATION CLASS
    // -------------------------------------------------------------------------------------------------------------

    // TODO: constructor to pass values to the storage (non-null initialization of vector)
    // TODO: check if view and clone are using same or new storage correctly

    struct TensorImpl
    {
        std::shared_ptr<Storage> storage_ = nullptr;
        ScalarType dtype_;
        Device device_;
        size_t offset_ = 0;                 /* Elements offset */
        std::vector<size_t> shape_;
        std::vector<size_t> strides_;
        size_t total_size_ = 0;             /* Number of elements */
        bool requires_grad_ = false;
        std::unique_ptr<AutogradMeta> autograd_meta_ = nullptr;

        // ---------------------------------------- HELPER FUNCTIONS ----------------------------------------
    
        void refresh_metadata();

        /* Helper: finds the actual memory address for logical indexing (useful in element-wise operations)*/
        size_t get_physical_offset(const std::vector<size_t>& indices) const;
    
        // ---------------------------------------- CONSTRUCTOR ----------------------------------------

        /* Impl from pointer to values */
        TensorImpl(const std::vector<size_t>& shape, ScalarType dtype, Device device, const void* src = nullptr);
        
        //TODO: requires methods for deep copy and shallow copy (same storage)


        // ---------------------------------------- GEOMETRIC FUNCTIONS ----------------------------------------

        bool is_contiguous() const;         /* Checks if the strides represent a contiguous representation of data */
        std::unique_ptr<TensorImpl> clone();
        std::unique_ptr<TensorImpl> contiguous() const;
        std::unique_ptr<TensorImpl> view(std::vector<size_t>& new_shape) const;
    };


    // -------------------------------------------------------------------------------------------------------------  
    //                                                  TENSOR CLASS
    // ------------------------------------------------------------------------------------------------------------- 

    /*
        TENSOR CLASS: wrapper of the TensorImpl class to be used as public API for the tensor library
        */
    class Tensor
    {
    private:

        // ---------------------------- IMPLEMENTATION ---------------------------- 
        std::unique_ptr<TensorImpl> pimpl_;
        Tensor(std::unique_ptr<TensorImpl> impl);   // Create tensor from existing implementation
    
    public:

        // ---------------------------------- ACCESSORS ---------------------------------- 

        TensorImpl* impl() const;
        const std::vector<size_t>& shape() const;
        const std::vector<size_t>& strides() const;
        ScalarType dtype() const;
        Device device() const;
        size_t size() const;
        size_t dims() const;
        bool requires_grad() const;
        void set_requires_grad(bool r) const;


        // ---------------------------------- CONSTRUCTORS ---------------------------------- 
        
        // ---------------------------------- empty constructors ---------------------------------- 


        // Empty tensor: no implementation
        Tensor();
        Tensor(const std::vector<size_t>& shape, ScalarType dtype = ScalarType::Float32, Device device = {DeviceType::CPU, 0});
        ~Tensor();
        Tensor(const Tensor& other);    /* Shallow copy (view) */
        Tensor clone() const;   /* Deep copy */

        // ---------------------------------- overloaded (?) constructors ---------------------------------- 

        /* Construct tensor from: shape (std::vector<size_t>), values(std::vector<T>) and device */
        template<typename T>
        Tensor(const std::vector<size_t>& shape, const std::vector<T>& values, Device device);

        // ---------------------------------- UTILITY FUNCTIONS ---------------------------------- 

        Tensor contiguous() const;
        bool is_contiguous() const;
        Tensor view(std::vector<size_t>& shape) const;
        Tensor reshape(std::vector<size_t>& shape) const;
    };



    // -------------------------------------------------------------------------------------------------------------  
    //                                                  BROADCASTING
    // ------------------------------------------------------------------------------------------------------------- 

    struct BroadcastInfo
    {
        std::vector<size_t> shape_;
        std::vector<size_t> strides_lhs_;    /* Virtual strides for lhs */
        std::vector<size_t> strides_rhs_;    /* Virtual strides for rhs */
    };
    
    inline std::vector<size_t> broadcast_shapes(const std::vector<size_t>& s1, const std::vector<size_t>& s2);
    // TODO: this has to be integrated in the dispatcher function interface
    inline BroadcastInfo get_broadcast_info(const TensorImpl& lhs, const TensorImpl& rhs);


    // -------------------------------------------------------------------------------------------------------------  
    //                                           OPERATION DISPATCHER INFRASTRUCTURE
    // ------------------------------------------------------------------------------------------------------------- 

    /* Need to know if working with a reduction (shape mismatch) */
    struct TensorIteratorConfig {
        std::vector<std::reference_wrapper<const Tensor>> inputs_;
        std::vector<std::reference_wrapper<Tensor>> outputs_;
        
        bool allow_cpu_cuda_mixing_ = false;
        
        bool is_reduction_ = false;
        bool promote_types_ = true;
        bool resize_outputs_ = false;   /* Allows iterator to allocate output memory */

        TensorIteratorConfig& add_input(const Tensor& t);
        TensorIteratorConfig& add_output(Tensor& t);
        TensorIteratorConfig& add_reduction(bool b);
    };

    struct OperandInfo {
        Tensor* tensor;
        ScalarType dtype;
        std::vector<size_t> strides;
        void* data;
    };

    class TensorIterator {
    private:

        std::vector<OperandInfo> operands_;
        Device common_device_;
        ScalarType common_dtype_;
        std::vector<size_t> shape_;

        void collapse_dims();   /*e.g. (100,100)->(10000) for speed if contiguous*/

    public:
        static TensorIterator build (const TensorIteratorConfig& config);

        // Metadata accessors
        Device device() const;
        ScalarType common_dtype() const;
        size_t num_elements() const;
        bool is_contiguous() const;

        static constexpr int64_t GRAIN_SIZE = 32768;    /* SIMD/CPU: grain size for multithreading */
        
        //  --------- kernel api ----------
        int ninputs() const;
        int noutputs() const;
        void* data_ptr(int arg) const;  /* Pointers to data for a specific operand index */
        bool is_trivial_1d() const; /* Optimization: true if all tensor are contiguous */
        size_t reduction_block_size() const; /* Reduction: returns number of elements in reduced dimenstions */

    };


    // -------------------------------------------------------------------------------------------------------------  
    //                                           BACKEND KERNEL INTERFACES
    // ------------------------------------------------------------------------------------------------------------- 
   
    struct CPUDevice {
        template<typename scalar_t, typename Op>
        static void launch_nullary(TensorIterator& iter, const Op& op);

        template<typename scalar_t, typename Op>
        static void launch_unary(TensorIterator& iter, const Op& op);
    
        template<typename scalar_t, typename Op>
        static void launch_binary(TensorIterator& iter, const Op& op);
        
        template<typename scalar_t, typename Op>
        static void launch_ternary(TensorIterator& iter, const Op& op);
        
        /* Requires two types scalar type and accumulator type (wider)*/
        template<typename scalar_t, typename acc_t, typename Op>
        static void launch_reduction(TensorIterator& iter, const Op& op, acc_t identity);

        template<typename scalar_t>
        static void launch_matmul(Tensor& out, const Tensor& lhs, const Tensor& rhs, const MatMulParams& params);
    };    

    #ifdef USE_CUDA
        struct CUDADevice {
            template<typename scalar_t, typename Op>
            static void launch_nullary(TensorIterator& iter, const Op& op);

            template<typename scalar_t, typename Op>
            static void launch_unary(TensorIterator& iter, const Op& op);
        
            template<typename scalar_t, typename Op>
            static void launch_binary(TensorIterator& iter, const Op& op);
            
            template<typename scalar_t, typename Op>
            static void launch_ternary(TensorIterator& iter, const Op& op);
            
            /* Requires two types scalar type and accumulator type (wider)*/
            template<typename scalar_t, typename acc_t, typename Op>
            static void launch_reduction(TensorIterator& iter, const Op& op, acc_t identity);

            template<typename scalar_t>
            static void launch_matmul(Tensor& out, const Tensor& lhs, const Tensor& rhs, const MatMulParams& params);
        }
    #endif


    // ------------------------------------------------ factory dispatcher --------------------------------------------
    // Useful for producing tensors (ones, zeros, arange, fill_)

    template<typename Op>
    struct NullaryElementwiseDispatcher {
        static void call(TensorIterator& iter, const std::string& name);
    };
    
    // ------------------------------------------------ comparison dispatcher --------------------------------------------
    // Technically binary but the output type is always bool

    template<typename Op>
    struct ComparisonDispatcher {
        static void call(TensorIterator& iter, const std::string& name);
    };

    // ------------------------------------------------ element-wise dispatcher --------------------------------------------

    /* Abstract backend dispatcher to be specialized for CPU and CUDA backends */
    template <typename Op>
    struct UnaryElementwiseDispatcher {
        static void call(TensorIterator& iter, const std::string& name);
    };

    template <typename Op>
    struct BinaryElementwiseDispatcher {
        static void call(TensorIterator& iter, const std::string& name);
    };

    template <typename Op>
    struct TernaryElementwiseDispatcher {
        static void call(TensorIterator& iter, const std::string& name);
    };

    // ----------------------------------------- linear algebra (matmul) dispatcher --------------------------------
    
    struct MatMulParams {
        bool trans_a = false;
        bool trans_b = false;
        double alpha = 1.0;
        double beta = 0.0;
    };
    
    struct MatMulDispatcher {
        static Tensor call(const Tensor& lhs, const Tensor& rhs, const MatMulParams& params = {});
    };

    std::vector<size_t> infer_matmul_shape(const std::vector<size_t>& lhs_shape,
                                           const std::vector<size_t>& rhs_shape,
                                           bool trans_a = false,
                                           bool trans_b = false);
    
    // ----------------------------------------- reduction dispatcher --------------------------------
    
    /* Reduction requires specific iterator configurations where the output has fewer dims */
    struct ReductionDispatcher {
        template<typename scalar_t, typename acc_t>
        static void call(TensorIterator& iter, const std::string& name, acc_t identity);
    };
    
    
    // -------------------------------------------------------------------------------------------------------------  
    //                                           OPERATOR DEFINITIONS (Functors)
    // -------------------------------------------------------------------------------------------------------------     
    
    
    // ------------------------------------------------ unary operators --------------------------------------------

    template <typename T>
    struct AbsFunctor {
        inline T operator()(T a) const;
    };

    template <typename T>
    struct ExpFunctor {
        inline T operator()(T a) const;
    };

    template <typename T>
    struct ReLUFunctor {
        inline T operator()(T a) const; // max(0, a)
    };


    // ------------------------------------------------ binary operators --------------------------------------------     

    template <typename T>
    struct AddFunctor {
        inline T operator()(T a, T b) const;
    };

    template <typename T>
    struct MulFunctor {
        inline T operator()(T a, T b) const;
    };


    // -------------------------------------------------------------------------------------------------------------  
    //                                         GLOBAL DISPATCH FUNCTIONS
    // ------------------------------------------------------------------------------------------------------------- 

    /**
     * Logic for Binary Operations:
     * 1. Check devices match.
     * 2. Determine promoted type.
     * 3. Construct TensorIterator.
     * 4. Select Backend (CPU/CUDA).
     * 5. Dispatch ScalarType (using DISPATCH_ALL_TYPES).
     */

    template <template <typename> class Op>
    Tensor unary_op_impl(const Tensor& lhs, const std::string& op_name)
    {
        Tensor result(lhs.shape(), lhs.dtype(), lhs.device());

        TensorIteratorConfig config;
        config.add_output(result).add_input(lhs);
        auto iter = TensorIterator::build(config);

        UnaryElementwiseDispatcher<Op>::call(iter, op_name);

        return result;
    }

    template <template <typename> class Op>
    Tensor binary_op_impl(const Tensor& lhs, const Tensor& rhs, const std::string& op_name)
    {
        if (lhs.device() != rhs.device()) {
            throw std::runtime_error(op_name + ": device mismatch");
        }
        auto out_shape = broadcast_shapes(lhs.shape(), rhs.shape());
        ScalarType out_dtype = promote_types(lhs.dtype(), rhs.dtype());
        Tensor result(out_shape, out_dtype, lhs.device());

        TensorIteratorConfig config;
        config.add_output(result).add_input(lhs).add_input(rhs);
        auto iter = TensorIterator::build(config);    
        
        BinaryElementwiseDispatcher<Op>::call(iter, op_name);
        
        return result;
    }

    // <!> for size > 2 the matmul operation is batched
    Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
        if( lhs.dims() < 2 || rhs.dims() < 2) {
            throw std::runtime_error("matmul: tensors must be at least 2D");
        }

        int64_t M = lhs.shape()[lhs.dims() - 2];
        int64_t K1 = lhs.shape()[lhs.dims() - 1];
        int64_t K2 = rhs.shape()[lhs.dims() - 2];
        int64_t N = rhs.shape()[lhs.dims() - 1];

        if (K1 != K2) {
            throw std::runtime_error("matmul: size_mismatch");
        }

        auto out_shape = infer_matmul_shape(lhs.shape(), rhs.shape());
        
        Tensor result(out_shape, promote_types(lhs.dtype(), rhs.dtype()), lhs.device());

        // TODO: check if to use (lhs, rhs, params)
        MatMulDispatcher::call(lhs, rhs);

        return result;
    }

    template <template <typename> class Op, typename acc_t>
    Tensor reduction_op_impl(const Tensor& lhs, std::vector<size_t> dims, bool keepdim,
                             acc_t identity, const std::string& name);


    // ----------------------- operations api ----------------------- 
    
    Tensor add(const Tensor& lhs, const Tensor& rhs)
    {
        return binary_op_impl<AddFunctor>(lhs, rhs, "add");
    }

    Tensor mul(const Tensor& lhs, const Tensor& rhs)
    {
        return binary_op_impl<MulFunctor>(lhs, rhs, "mul");
    }
    
    Tensor ReLU(const Tensor& lhs)
    {
        return unary_op_impl<ReLUFunctor>(lhs, "relu");
    }
    
    // reduction op
    Tensor sum(const Tensor& lhs, std::vector<size_t> dims, bool keepdim) {
        return reduction_op_impl<AddFunctor>(lhs, dims, keepdim, 0.0, "sum");
    }

    Tensor mean(const Tensor& lhs, std::vector<size_t> dims, bool keepdim)
    {
        Tensor result = sum(lhs, dims, keepdim);

        double count = calculate_reduction_count(lhs, dims);
        return result / count;
    }
    // ----------------------- operators oveloading ----------------------- 
    
    inline Tensor operator+(const Tensor& lhs, const Tensor& rhs)
    {
        return add(lhs,rhs);
    }


    // TODO: add overload with scalars (Tensor + scalar, Tensor * scalar)

} //namespace tensor