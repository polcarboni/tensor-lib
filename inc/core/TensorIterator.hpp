#pragma once
#include <vector>
#include <memory>
#include "Types.hpp"
#include "TensorImpl.hpp"

namespace tensor
{
    // struct ScalarType;
    // struct Device;
    // class TensorImpl;
    struct FillOpBase;

    enum class IterationKind { ELEMENT_WISE, REDUCTION, MATMUL, SCALAR, COPY };
    enum class Direction { FORWARD, BACKWARD };

    class TensorIterator {
    private: 
        // std::vector<std::shared_ptr<TensorImpl>> inputs_;
        // std::vector<std::shared_ptr<TensorImpl>> outputs_;

        // Better version: avoid copy
        std::vector<TensorImpl*> inputs_;                               /* non-const for allowing inplace operations */
        std::vector<TensorImpl*> outputs_;

        std::vector<std::unique_ptr<TensorImpl>> materialized_inputs_;  /* Type casted input copies */
        std::unique_ptr<TensorImpl> nullary_output_;

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
        ScalarType compute_common_dtype_()
        {
            if (inputs_.empty() && outputs_.empty()) {
                return ScalarType::Float32;
            }

            ScalarType promoted = ScalarType::Bool;
            
            for (auto* input : inputs_)
            {
                if(!input) continue;
                promoted = promote_types(promoted, input->get_dtype());
            }
            
            for (auto* output : outputs_)
            {
                if(!output) continue;
                promoted = promote_types(promoted, output->get_dtype());
            }
            
            return promoted;
        }

        /**
         * Checks if the operands are on the same device for computing operations.
         * Cross-device computation is not automatically allowed  (requires previous manual movement). 
         */
        Device compute_common_device_()
        {
            Device common_dev{};
            bool device_initialized = false;
            
            for (auto* output: outputs_)
            {
                if(!output) continue;

                if(!device_initialized) {
                    common_dev = output->get_device();
                    device_initialized = true;
                } else {
                    if (output->get_device() != common_dev) {
                        throw std::runtime_error("TensorIterator: Expected tensor to be on the same device");
                    }
                }
            }

            for (auto* input : inputs_)
            {
                if (!input) continue;

                if(!device_initialized) {
                    common_dev = input->get_device();
                    device_initialized = true;
                }
                else {
                    if (input->get_device() != common_dev) {
                        throw std::runtime_error("TensorIterator: Expected tensor to be on the same device");
                    }
                }
            }

            return common_dev;
        }
        
        /**
         * Checks if all the tensor operands are contiguous to allow the use of faster computation paths.
         */
        bool check_contiguous_()
        {
            for (auto* output : outputs_) {
                if(output && !output->is_contiguous()) {
                    return false;
                }
            }

            for (auto* input: inputs_) {
                if (input && !input->is_contiguous()) {
                        return false;
                }
            }
            return true;
        }
      

        /**
         * Checks the output tensor grad requirements. If not specified, true if at least an input tensor
         * has requires_grad as true.
         */
        bool compute_requires_grad_() {

            for (auto* output : outputs_) {
                if(output && output->requires_grad()) {
                    return true;
                }
            }

            for (auto* input: inputs_) {
                if (input && input->requires_grad()) {
                    if (common_dtype_ == ScalarType::Float32 || common_dtype_ == ScalarType::Float64)
                        return true;
                }
            }
            return false;
        }

        
        /**
         * Checks instantiation correctness: correct number of operands, operation existence, type correctness
         * and other ...
         * Throws: invalid shapes for broadcasting, mismatch of broadcasted shape with output shape if provided
         */
        template<typename Op>
        void validate_inputs_metadata_()
        {
            
            if (inputs_.size() != Op::num_inputs()) {
                throw std::runtime_error("Wrong number of inputs: " + std::to_string(inputs_.size()) +
                                         ", expected: " + std::to_string(Op::num_inputs()));
            }

            common_device_ = compute_common_device_();
            common_dtype_  = compute_common_dtype_();
            common_requires_grad_ = compute_requires_grad_();
        }

        
        // -------------------------------------------------------------------------------------------------------------  
        //                                                SHAPES BROADCASTING
        // ------------------------------------------------------------------------------------------------------------- 


        // TODO: consider separating broadcasting and validation
        template <typename Op>
        std::vector<size_t> broadcast_shapes_()
        {
            if constexpr      (Op::iter_kind() == IterationKind::ELEMENT_WISE)  return broadcast_shapes_elementwise_<Op>();
            else if constexpr (Op::iter_kind() == IterationKind::REDUCTION)     return broadcast_shapes_reduction_<Op>();
            else if constexpr (Op::iter_kind() == IterationKind::MATMUL)        return broadcast_shapes_matmul_<Op>();
            else if constexpr (Op::iter_kind() == IterationKind::SCALAR)        return broadcast_shapes_scalar_<Op>();
            else if constexpr (Op::iter_kind() == IterationKind::COPY)          return broadcast_shapes_copy_<Op>();
        }

        /**
         * Defines the common resulting shape for all the input tensors. Provide different shape computation
         * paths based on the required operation types.
         */
        template <typename Op>
        std::vector<size_t> broadcast_shapes_elementwise_()
        {
            if (inputs_.empty()) {
                return {};
            }

            size_t max_ndim = 0;
            for (auto* input: inputs_) {
                if (!input) continue;
                max_ndim = std::max(max_ndim, input->get_shape().size());
            }

            std::vector<size_t> result_shape(max_ndim, 1);

            for (auto* input: inputs_)
            {
                const auto& current_shape = input->get_shape();
                size_t current_ndim = current_shape.size();

                for (size_t i = 0; i < current_ndim; ++i)
                {
                    size_t result_idx = max_ndim - 1 - i;
                    size_t input_idx = current_ndim - 1 - i;

                    size_t input_dim_size = current_shape[input_idx];
                    size_t result_dim_size = result_shape[result_idx];

                    if (result_dim_size == 1) {
                        result_shape[result_idx] = input_dim_size;
                    }
                    else if (input_dim_size != 1 && input_dim_size != result_dim_size) {
                        throw std::runtime_error("Error: shapes could not be broadcast together: " +
                                                 std::to_string(input_dim_size) + ", " +
                                                 std::to_string(result_dim_size));
                    }
                }
            }

            // // Validate the output shape
            // if (output_) {
            //     const auto& output_shape = output_->get_shape();

            //     if (output_shape.size() != result_shape.size()) {
            //         throw std::runtime_error("...Shape dimensionality mismatch...");
            //     }

            //     for (size_t i = 0; i < result_shape.size(); ++i) {
            //         if (output_shape[i] != result_shape[i]) {
            //             throw std::runtime_error("...Mismatch in one dimension...");
            //         }
            //     }
            // }
            return result_shape;
        }

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
        std::vector<std::vector<size_t>> compute_broadcast_strides_()
        {
            if      constexpr (Op::iter_kind() == IterationKind::ELEMENT_WISE) return compute_strides_elementwise_<Op>();
            else if constexpr (Op::iter_kind() == IterationKind::REDUCTION)    return compute_strides_reduction_<Op>();
            else if constexpr (Op::iter_kind() == IterationKind::MATMUL)       return compute_strides_matmul_<Op>();
            else if constexpr (Op::iter_kind() == IterationKind::SCALAR)       return compute_strides_scalar_<Op>();
            else if constexpr (Op::iter_kind() == IterationKind::COPY)         return compute_strides_copy_<Op>();
        }
        

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
        std::vector<std::vector<size_t>> compute_strides_elementwise_()
        {
            std::vector<std::vector<size_t>> strides;

            if constexpr (Op::get_direction() == Direction::FORWARD) {
                
                strides.reserve(inputs_.size() + outputs_.size());

                for (auto* operand : inputs_) {
                    std::vector<size_t> operand_strides;
                    size_t ndim = output_shape_.size();

                    const auto& op_shape = operand->get_shape();
                    const auto& op_strides = operand->get_strides();

                    if (operand->is_contiguous()) {
                        operand_strides.resize(ndim, 0);
                        size_t stride = 1;
                        for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
                            operand_strides[i] = stride;
                            stride *= output_shape_[i];
                        }
                    } else {
                        // strides from operand metadata and
                        // pad with 0s left dimensions added by broadcasting

                        operand_strides.resize(ndim, 0);
                        size_t ndim_op = op_strides.size();
                        size_t offset = ndim - ndim_op;     // padding

                        for (size_t i = 0; i < ndim_op; ++i) {
                            operand_strides[offset + i] = op_strides[i];
                        }
                    }

                    // Broadcasting rule: operand size = 1 in dimension -> stride = 0
                    size_t ndim_op = op_shape.size();
                    size_t offset = ndim - ndim_op;

                    for (size_t i = 0; i < ndim; ++i) {
                        // left padded dimensions are implicitly size-1
                        bool is_prepended = (i < offset);
                        bool is_size_one = !is_prepended && (op_shape[i - offset] == 1);
                        if(!is_prepended || is_size_one) {
                            operand_strides[i] = 0;
                        }
                    }
                    strides.push_back(std::move(operand_strides));
                }

                for (auto* operand : outputs_) {
                    std::vector<size_t> operand_strides;
                    size_t ndim = broadcasted_shape_.size();

                    if(operand->is_contiguous()) {
                        // output never broadcast-reduced
                        operand_strides.resize(ndim, 0);
                        size_t stride = 1;
                        for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
                            operand_strides[i] = stride;
                            stride *= output_shape_[i];
                        }
                    } else {
                        // Non contiguous output: use actual stride directly
                        const auto& op_strides = operand->get_strides();
                        size_t ndim_op = op_strides.size();
                        operand_strides.resize(ndim, 0);
                        size_t offset = ndim - ndim_op;

                        for (size_t i = 0; i < ndim_op; ++i) {
                            operand_strides[offset + i] = op_strides[i];
                        }
                    }

                    strides.push_back(std::move(operand_strides));
                }
            }

            else if (Op::get_direction() == Direction::BACKWARD) {
                throw std::runtime_error("Compute element-wise strides: backward is NOT IMPLEMENTED");
            }

            return strides;
        }
        
        template <typename Op>
        std::vector<std::vector<size_t>> compute_strides_reduction_(){
            if (Op::get_direction() == Direction::FORWARD) {}
            else if (Op::get_direction() == Direction::BACKWARD) {}
        }
        
        template <typename Op>
        std::vector<std::vector<size_t>> compute_strides_matmul_(){
            if (Op::get_direction() == Direction::FORWARD) {}
            else if (Op::get_direction() == Direction::BACKWARD) {}
        }
        
        template <typename Op>
        std::vector<std::vector<size_t>> compute_strides_scalar_(){
            if (Op::get_direction() == Direction::FORWARD) {}
            else if (Op::get_direction() == Direction::BACKWARD) {}
        }
        
        template <typename Op>
        std::vector<std::vector<size_t>> compute_strides_copy_(){
            if (Op::get_direction() == Direction::FORWARD) {}
            else if (Op::get_direction() == Direction::BACKWARD) {}
        }

        // -------------------------------------------------------------------------------------------------------------  
        //                                                  TYPES MATERIALIZATION
        // ------------------------------------------------------------------------------------------------------------- 

        /**
         * Calls the type casting operations for tensor operators with dtype different from iterator.common_dtype_ 
         */
        void materialize_inputs_()
        {
            for (size_t i = 0; i < inputs_.size(); ++i) {
                if (!inputs_[i] || inputs_[i]->get_dtype() == common_dtype_) continue;
                    // This will produce a nested iterator call
                    materialized_inputs_.push_back(inputs_[i]->to_dtype(common_dtype_));
                    inputs_[i] = materialized_inputs_.back().get();
            }
        }


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

        // --------------------------- DISPATCHER --------------------------- 

        void add_input(TensorImpl* tensor)
        {
            inputs_.push_back(tensor);
        }
        
        void add_output(TensorImpl* tensor)
        {
            outputs_.push_back(tensor);
        }
        
        /**
         * Checks correctenss by validating the inputs, computing the broadcasted shape and broadcasted strides
         * used by the kernels for accessing the tensor Storage elements.  
         */
        template <typename Op>
        void build(const std::vector<size_t>& shape = {},
                   const ScalarType cast_type = ScalarType::EMPTY)
        {
            validate_inputs_metadata_<Op>();

            // Calls tensorImpl.to_dtype() for all input tensors with type different than common_dtype_
            // and changes the inputs_ vector inplace.
            materialize_inputs_();

            if (Op::get_direction() == Direction::FORWARD)
            {
                if (outputs_.size() > 1) {
                    throw std::runtime_error("Forward operation expects at most 1 output");
                }

                // UNARY CASTING OPERATION: uses the provided cast_type argument
                // TODO: add check also on the operation template
                if (cast_type != ScalarType::EMPTY) {
                    if(shape.empty()) {
                        throw std::runtime_error("Cannot cast an empty tensor");
                    }
                    common_dtype_ = cast_type;
                }

                // FILLING OPERATION: uses the provided shape argument (no broadcasting)
                // TODO: add check also on the operation template
                // TODO: this should also use the dtype
                if constexpr (std::is_base_of_v<FillOpBase, Op>) {
                    if (shape.empty()) {
                        throw std::runtime_error("Fill operation requires an explicit output shape");
                    }
                    output_shape_ = shape;
                
                } else {
                    broadcasted_shape_ = broadcast_shapes_<Op>();
                    output_shape_ = broadcasted_shape_;
                }

                if (outputs_.empty()) {
                    outputs_.push_back(nullptr);
                }

                if (!outputs_[0]) {
                    nullary_output_ = std::make_unique<TensorImpl>(output_shape_, common_dtype_, common_device_, common_requires_grad_);
                    outputs_[0] = nullary_output_.get();
                }

                if(common_requires_grad_) {
                    // Initialize grads: outputs_[0]->init_autograd_meta()
                    // Store input shapes in output_tensor->autograd_meta_->grad_fn_->input_shapes_;
                    //      required for backward pass
                }
            }
            
            else if (Op::get_direction() == Direction::BACKWARD) {

                if (inputs_.size() != Op::num_outputs()) {
                    throw std::runtime_error("Backward ERROR ....");
                }

                if (outputs_.size() != Op::num_inputs()) {
                    throw std::runtime_error("Expected a grad tensor per forward input.");
                }

                for (size_t i = 0; i < Op::num_inputs(); i++) {

                    if (!outputs_[i]) {
                        throw std::runtime_error("Backward pass expects preallocatd tensors ...");
                    } 
                    
                    if (!outputs_[i]->requires_grad()) {
                        throw std::runtime_error("Backward: tensor has no requires_grad_ ...");
                    }
                }
            }

            common_is_contiguous_ = check_contiguous_();
            broadcasted_strides_ = compute_broadcast_strides_<Op>();
        }

        
        std::vector<TensorImpl*> get_outputs()
        {
            return outputs_;
        }

        // --------------------------- KERNEL ACCESSORS ---------------------------
        
        // accessors should cast to common_dtype_
        
        void* input_data(int idx)
        {
            if (idx < 0 || static_cast<size_t>(idx) >= inputs_.size())
                throw std::out_of_range("input_data: out_of range");
            return inputs_[idx]->data_ptr();
        }
        const void* input_data(int idx) const;
        
        void* output_data(int idx)
        {
            if (idx < 0 || idx >= static_cast<int>(outputs_.size()))
            throw std::out_of_range("output_data: index out of range");
            return outputs_[idx]->data_ptr();
        }
        const void* output_data(int idx) const;
        
        void* output_grad_data(int idx);
        const void* output_grad_data(int idx) const;
        
        // --------------------------- TYPED KERNEL ACCESSORS ---------------------------

        template <typename T>
        T* input_ptr(int idx)
        {
            return static_cast<T*>(input_data(idx));
        }

        template <typename T>
        T* output_ptr(int idx = 0)
        {
            return static_cast<T*>(output_data(idx));
        }
    };

} //namespace tensor