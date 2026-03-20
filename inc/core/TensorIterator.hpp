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

    enum class IterationKind { ELEMENT_WISE, REDUCTION, MATMUL, COPY };
    enum class Direction { FORWARD, BACKWARD };

    class TensorIterator {
    private: 

        std::vector<TensorImpl*> inputs_;                               /* pointers to input tensors, updated using add_input( ) */
        std::vector<TensorImpl*> outputs_;                              /* pointers to output tensors, updated using add_output( ) */
        std::vector<std::vector<size_t>> output_shapes_;                /* original shapes of provided inplace output tensors, updated using add_output( ) */

        std::vector<std::unique_ptr<TensorImpl>> materialized_inputs_;  /* Type casted input copies, created with potiner added to outputs_ */
        std::unique_ptr<TensorImpl> nullary_output_;                    /* Synthesized output tensor (for non-inplace operations) */
        bool inplace_ = false;                                          /* STILL UNUSED */

        ScalarType common_dtype_;                                       /* true if operands have same dtype_ (cast type if not) */
        Device     common_device_;                                      /* true if operands are on same device (hard requirement) */
        bool       common_is_contiguous_ = false;                       /* true if all operands are contiguous */
        bool       common_requires_grad_ = false;                       /* true if all tensors require grads */

        std::vector<std::vector<size_t>> broadcasted_shapes_;           /* SHOULD have both inputs and outputs shapes*/
        std::vector<std::vector<size_t>> broadcasted_strides_;          /* iterator space strides */
        bool is_broadcasted_ = false;                                   /* if non active the operation has not used broadcast and can use fast path */
        
        size_t numel_;                                                  /* Number of elements of the iterator space */
        size_t ndim_;                                                   /* Rank of the iterator space */


        // -------- Scalar operations data members -------- 

        bool scalar_ = false;


        // -------- Reduction operations data members -------- 

        std::optional<std::vector<size_t>> reduction_axes_ = std::nullopt;
        bool keepdims_ = false;
        int num_reduced_axes_ = 0;
        bool contiguous_along_reduced_axes_ = false;
        std::optional<std::vector<bool>> is_reduced_dim_ = std::nullopt;
        std::optional<std::vector<bool>> compute_is_reduced_dim_(std::optional<std::vector<size_t>>& reduced_axes, size_t ndim);


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

        bool compute_contiguous_along_reduced_axes_();
        int count_num_reduced_axes_(TensorImpl* input);
        
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
        void check_shapes_copy_();


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
        std::vector<std::vector<size_t>> compute_strides_copy_();


        // -------------------------------------------------- TYPES MATERIALIZATION --------------------------------------------------

        /**
         * Calls the type casting operations for tensor operators with dtype different from iterator.common_dtype_ 
         */
        void materialize_inputs_();
        

        // -------------------------------------------------- DIMENSIONS COALESCING --------------------------------------------------

        /**
         * Called by the coalesce_dimensions_( ) private method.
         * 
         * Computes the possibility of each dimension to be merged with the following one.
         * 
         * TODO: fix the API since the reduction uses mixed explicit and implicit members.
         */
        template <typename Op>
        std::vector<bool> compute_merge_decision_(std::vector<std::vector<size_t>>& shapes,
                                                  std::vector<std::vector<size_t>>& strides);

        template <typename Op>
        std::vector<bool> compute_merge_decision_elementwise_(std::vector<std::vector<size_t>>& shapes,
                                                              std::vector<std::vector<size_t>>& strides);

        template <typename Op>
        std::vector<bool> compute_merge_decision_reduction_(std::vector<std::vector<size_t>>& shapes,
                                                            std::vector<std::vector<size_t>>& strides);

        template <typename Op>
        std::vector<bool> compute_merge_decision_matmul_(std::vector<std::vector<size_t>>& shapes,
                                                         std::vector<std::vector<size_t>>& strides);

        template <typename Op>
        std::vector<bool> compute_merge_decision_copy_(std::vector<std::vector<size_t>>& shapes,
                                                       std::vector<std::vector<size_t>>& strides);

                                                    
        /**
         * Called by the coalesce_dimensions_( ) private method.
         * 
         * Applies a merge decision to a shape vector collapsing mergeable dimensions into a
         * single dimension whose size is the product of the merged ones.
         */
        std::vector<size_t> apply_merge_to_shape_(std::vector<size_t>& shape,
                                                  std::vector<bool>& merge_decision);
        
        /**
         * Called by the coalesce_dimensions_( ) private method.
         * 
         * Applies a merge decision to all oeprands' stride vector, collapsing mergeable
         * dimensions into a single stride entry.
         * 
         * Strides are defined and processed per-operand, differently from the shape.
         */                                          
        std::vector<std::vector<size_t>> apply_merge_to_strides_(std::vector<std::vector<size_t>>& strides,
                                                                 std::vector<bool>& merge_decision);

        /**
         * Attempts to coalesce dimensions into fewer and larger dimensions, computes the merge
         * decision based on the provided shapes and strides. Modifies the inputs inplace.
         * 
         * Return true if coalescing was applied, false otherwise.
         */
        template <typename Op>
        bool coalesce_dimensions_(std::vector<std::vector<size_t>>& shapes,
                                  std::vector<std::vector<size_t>>& strides);


        

        /**
         * Compute the number of iterator space elements based on the type of operation:
         *  - ELEMENT_WISE: number of elements of the output tensor.
         *  - REDUCTION: still not implemented.
         *  - MATMUL: still not implemented.
         */
        template <typename Op>
        size_t compute_numel_(const std::vector<std::vector<size_t>>& broadcasted_strides);
        
        /**
         * Compute the number of dimensions of the iterator space:
         *  - ELEMENT_WISE: rank of any broadcasted operand.
         *  - REDUCTION: still not implemented.
         *  - MATMUL: still not implemented.
         */
        template <typename Op>
        size_t compute_ndim_(const std::vector<std::vector<size_t>>& broadcasted_strides);
        


        // =============================================================================================================  
        //                                                PUBLIC INTERFACES
        // =============================================================================================================
    
    public:
        
        TensorIterator() = default;
        TensorIterator(const TensorIterator&) = delete;
        TensorIterator& operator=(const TensorIterator&) = delete;
        TensorIterator(TensorIterator&&) = default;
        TensorIterator& operator=(TensorIterator&&) = default;

        // -------------------------------------------------- GETTERS --------------------------------------------------
        
        std::vector<TensorImpl*> get_outputs();
        
        bool get_inplace();
        ScalarType get_common_dtype();
        Device get_common_device();
        bool get_common_is_contiguous();
        bool get_common_requires_grad();
        bool get_scalar();
        bool get_keepdims() const;
        bool get_is_broadcasted() const;

        bool get_contiguous_along_reduced_axes() const;
        int  get_num_reduced_axes() const;
        std::optional<std::vector<bool>> get_is_reduced_dim() const;
        std::optional<std::vector<size_t>> get_reduction_axes() const;
        
        void set_inplace(bool val);
        void set_reduction_axes(const std::optional<std::vector<size_t>> axes);
        void set_keepdims(bool keepdims);
        void set_scalar(bool scalar);

        ScalarType get_input_dtype() const; // for casting operation

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


        /**
         * Returns the number of dimensions of the iteration space.
         */
        size_t get_ndim() const;

        /**
         * Return the total number of elements in the iteration space.
         */
        size_t get_numel() const;

        /**
         * Return the shape of the iteration space
         */
        const std::vector<size_t>& get_shape() const;

        /**
         * Returns the strides of a specific operand (input or output).
         */
        const std::vector<size_t>& get_strides(int arg_idx) const;

        /**
         * Returns the stride for a specific operand at a specific dimension.
         */
        size_t get_stride(int arg_idx, int dim_idx) const;
    };

} //namespace tensor