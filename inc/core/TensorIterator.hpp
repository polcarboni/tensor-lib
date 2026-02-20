#pragma once
#include <vector>
#include <memory>
#include "core/Types.hpp"
// #include "core/TensorImpl.hpp"

namespace tensor
{
    /* Forward declarations */
    struct TensorImpl;
    // struct FillOpBase;

    /* Itertot types classes */

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

        std::vector<size_t>              broadcasted_shape_;
        std::vector<size_t>              output_shape_;
        std::vector<std::vector<size_t>> broadcasted_strides_;


        // -------------------------------------------------------------------------------------------------------------  
        //                                                METADATA VALIDATION
        // -------------------------------------------------------------------------------------------------------------

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

        
        // -------------------------------------------------------------------------------------------------------------  
        //                                                SHAPES BROADCASTING
        // ------------------------------------------------------------------------------------------------------------- 


        // TODO: consider separating broadcasting and validation
        template <typename Op>
        std::vector<size_t> broadcast_shapes_();

        /**
         * Defines the common resulting shape for all the input tensors. Provide different shape computation
         * paths based on the required operation types.
         */
        template <typename Op>
        std::vector<size_t> broadcast_shapes_elementwise_();

        template <typename Op>
        std::vector<size_t> broadcast_shapes_reduction_();
        
        template <typename Op>
        std::vector<size_t> broadcast_shapes_matmul_();

        template <typename Op>
        std::vector<size_t> broadcast_shapes_scalar_();
        
        template <typename Op>
        std::vector<size_t> broadcast_shapes_copy_();


        // -------------------------------------------------------------------------------------------------------------  
        //                                                  STRIDES BROADCASTING
        // ------------------------------------------------------------------------------------------------------------- 

        template <typename Op>
        std::vector<std::vector<size_t>> compute_broadcast_strides_();
        
        /**
         * TODO: this function was completely implemented via LLM (it is wrong).
         * 
         * TODO: contiguous operands can use the strides member instead of computing it again.
         * Other als omight already have the strides.
         * 
         * Strides can be hoever changed due to the broadcasting logic.
         * TODO: provide fast path for operations that do not require it: filling, same size pointwise, other ...
         * 
         * Not sure how the forward and backward should be different. Maybe for this case
         * (element wise operations) can be the same, but not for the other ones.
         */
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

        // -------------------------------------------------------------------------------------------------------------  
        //                                                  TYPES MATERIALIZATION
        // ------------------------------------------------------------------------------------------------------------- 

        /**
         * Calls the type casting operations for tensor operators with dtype different from iterator.common_dtype_ 
         */
        void materialize_inputs_();


        // =============================================================================================================  
        // =============================================================================================================  
        //                                                PUBLIC INTERFACES
        // =============================================================================================================
        // =============================================================================================================
    
    public:
        
        TensorIterator() = default;
        TensorIterator(const TensorIterator&) = delete;
        TensorIterator& operator=(const TensorIterator&) = delete;
        TensorIterator(TensorIterator&&) = default;
        TensorIterator& operator=(TensorIterator&&) = default;

        bool get_inplace();
        void set_inplace(bool val);


        // --------------------------- DISPATCHER --------------------------- 

        void add_input(TensorImpl* tensor);
        void add_output(TensorImpl* tensor);
        
        /**
         * Checks correctenss by validating the inputs, computing the broadcasted shape and broadcasted strides
         * used by the kernels for accessing the tensor Storage elements.  
         */
        template <typename Op>
        void build(const std::vector<size_t>& shape = {},
                   const ScalarType cast_type = ScalarType::EMPTY);
        
        std::vector<TensorImpl*> get_outputs();


        // --------------------------- KERNEL ACCESSORS ---------------------------
        
        void* input_data(int idx);
        const void* input_data(int idx) const;
        
        void* output_data(int idx);
        const void* output_data(int idx) const;
        
        void* output_grad_data(int idx);
        const void* output_grad_data(int idx) const;
        

        // --------------------------- TYPED KERNEL ACCESSORS ---------------------------

        template <typename T>
        T* input_ptr(int idx);

        template <typename T>
        T* output_ptr(int idx = 0);
    };

} //namespace tensor