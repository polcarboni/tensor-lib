#pragma once
#include "core/Types.hpp"
#include <vector>
#include <memory>
#include <optional>

namespace tensor
{
    /* Forward declarations */
    
    struct TensorImpl;

    /* Iterator types classes */

    enum class IterationKind { ELEMENT_WISE, REDUCTION, MATMUL, SCALAR, COPY };
    enum class Direction { FORWARD, BACKWARD };

    class TensorIterator {
    private: 

        std::vector<TensorImpl*> inputs_;                               /* non-const for allowing inplace operations */
        std::vector<TensorImpl*> outputs_;

        std::vector<std::unique_ptr<TensorImpl>> materialized_inputs_;  /* Type casted input copies */
        std::unique_ptr<TensorImpl> nullary_output_;
        bool inplace_ = false;

        ScalarType common_dtype_;
        Device     common_device_;
        bool       common_is_contiguous_ = false;
        bool       common_requires_grad_ = false;

        std::vector<std::vector<size_t>> broadcasted_shapes_;
        std::vector<std::vector<size_t>> output_shapes_;

        bool is_broadcasted_ = false;                                    /* if non active the operation has not used broadcast and can use fast path */
        std::vector<std::vector<size_t>> broadcasted_strides_;


        // -------- Reduction operations data members -------- 

        std::optional<std::vector<size_t>> reduction_axes_;
        bool keepdims_ = false;


        // ---------- Matmul operations data members ----------

        size_t m_ = 0;
        size_t n_ = 0;
        size_t k_ = 0;

        bool trans_a_ = false;
        bool trans_b_ = false;

        int64_t batch_stride_a_ = 0;
        int64_t batch_stride_b_ = 0;
        int64_t batch_stride_out_ = 0;


        // -------------------------------------------------- METADATA VALIDATION --------------------------------------------------

        /**
         * Compute the output type using the type promotion rules and the operands ScalarType.
         */
        ScalarType compute_common_dtype_();

        /**
         * Checks if the operands are on the same device for computing operations.
         * Cross-device computation is not automatically allowed  (requires previous manual movement). 
         */
        Device compute_common_device_();

        /**
         * Checks if all the tensor operands are contiguous to allow the use of faster computation paths.
         */
        bool check_contiguous_();

        /**
         * Checks the output tensor grad requirements. If not specified, true if at least an input tensor
         * has requires_grad as true.
         */
        bool compute_requires_grad_();
        
        /**
         * Checks instantiation correctness: correct number of operands, operation existence, type correctness
         * and other ...
         * Throws: invalid shapes for broadcasting, mismatch of broadcasted shape with output shape if provided
         */
        template<typename Op>
        void validate_inputs_metadata_();

        
        // -------------------------------------------------- SHAPES BROADCASTING --------------------------------------------------

        template <typename Op>
        std::vector<std::vector<size_t>> broadcast_shapes_();


        template <typename Op>
        std::vector<std::vector<size_t>> broadcast_shapes_elementwise_();

        template <typename Op>
        std::vector<std::vector<size_t>> broadcast_shapes_reduction_();
        
        template <typename Op>
        std::vector<std::vector<size_t>> broadcast_shapes_matmul_();

        template <typename Op>
        std::vector<std::vector<size_t>> broadcast_shapes_scalar_();
        
        template <typename Op>
        std::vector<std::vector<size_t>> broadcast_shapes_copy_();


        // -------------------------------------------------- STRIDES BROADCASTING --------------------------------------------------

        template <typename Op>
        std::vector<std::vector<size_t>> compute_broadcast_strides_();
        

        template <typename Op>
        std::vector<std::vector<size_t>> compute_strides_elementwise_();
        
        template <typename Op>
        std::vector<std::vector<size_t>> compute_strides_reduction_();
        
        template <typename Op>
        std::vector<std::vector<size_t>> compute_strides_matmul_();
        
        template <typename Op>
        std::vector<std::vector<size_t>> compute_strides_scalar_();
        
        template <typename Op>
        std::vector<std::vector<size_t>> compute_strides_copy_();


        // -------------------------------------------------- TYPES MATERIALIZATION --------------------------------------------------

        /**
         * Calls the type casting operations for tensor operators with dtype different from iterator.common_dtype_ 
         */
        void materialize_inputs_();


        // =============================================================================================================  
        //                                                PUBLIC INTERFACES
        // =============================================================================================================
    
    public:
        
        TensorIterator();
        TensorIterator(const TensorIterator&);
        TensorIterator& operator=(const TensorIterator&);
        TensorIterator(TensorIterator&&);
        TensorIterator& operator=(TensorIterator&&);


        // -------------------------------------------------- GETTERS --------------------------------------------------
        
        std::vector<TensorImpl*> get_outputs();
        
        bool get_inplace();
        ScalarType get_common_dtype();
        Device get_common_device();
        bool get_common_is_contiguous();
        bool get_common_requires_grad();
        
        void set_inplace(bool val);
        void set_reduction_axes(const std::optional<std::vector<size_t>> axes);
        void set_keepdims(bool keepdims);

        // -------------------------------------------------- DISPATCHER INTERFACES -------------------------------------------------- 

        void add_input(TensorImpl* tensor);
        void add_output(TensorImpl* tensor);
        
        /**
         * Checks correctenss by validating the inputs, computing the broadcasted shape and broadcasted strides
         * used by the kernels for accessing the tensor Storage elements.  
         */
        template <typename Op>
        void build(const ScalarType cast_type = ScalarType::EMPTY);


        // -------------------------------------------------- KERNEL INTERFACES --------------------------------------------------
        
        void* input_data(int idx);
        const void* input_data(int idx) const;
        
        void* output_data(int idx);
        const void* output_data(int idx) const;
        
        void* output_grad_data(int idx);
        const void* output_grad_data(int idx) const;
    

        template <typename T>
        T* input_ptr(int idx);

        template <typename T>
        T* output_ptr(int idx = 0);
    };

} //namespace tensor