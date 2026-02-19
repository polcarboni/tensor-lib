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
        std::vector<std::shared_ptr<TensorImpl>> inputs_;
        // std::shared_ptr<TensorImpl> output_;
        std::vector<std::shared_ptr<TensorImpl>> outputs_;
        // std::vector<std::shared_ptr<TensorImpl>> output_grads_;
        // NO: grads are sotred in outputs_[i]->autograd_meta_->grad_

        ScalarType common_dtype_;
        Device common_device_;
        bool common_is_contiguous_ = false;
        bool common_requires_grad_ = false;

        std::vector<size_t> broadcasted_shape_;
        std::vector<size_t> output_shape_;
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
            
            for (const auto& input : inputs_)
            {
                if(!input) continue;
                promoted = promote_types(promoted, input->get_dtype());
            }
            
            for (const auto& output : outputs_)
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
            
            for (const auto& output: outputs_)
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

            for (const auto& input : inputs_)
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
        bool check_contiguous_();
      

        /**
         * Checks the output tensor grad requirements. If not specified, true if at least an input tensor
         * has requires_grad as true.
         */
        bool compute_requires_grad_() {

            for (const auto& output : outputs_) {
                if(output && output->requires_grad()) {
                    return true;
                }
            }

            for (const auto& input: inputs_) {
                if (input->requires_grad()) {
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
            if constexpr (Op::iter_kind() == IterationKind::ELEMENT_WISE) {
                // if (inputs_.empty()) {
                //     return output_ ? output_->get_shape() : throw std::runtime_error("NOT SURE CHECK AGAIN")
                // } else {
                //     return broadcast_shapes_elementwise_();
                // }
                return broadcast_shapes_elementwise();
            } else if constexpr (Op::iter_kind() == IterationKind::REDUCTION) {
                return broadcast_shapes_reduction<Op>();
            } else if constexpr (Op::iter_kind() == IterationKind::MATMUL) {
                return broadcast_shapes_matmul_<Op>();
            } if constexpr (Op::iter_kind() == IterationKind::SCALAR) {
                return broadcast_shapes_scalar_();
            } if constexpr (Op::iter_kind() == IterationKind::COPY) {
                return broadcast_shapes_copy_();
            }
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
            for (const auto& input: inputs_) {
                if (!input) continue;
                max_ndim = std::max(max_ndim, input->get_shape().size());
            }

            std::vector<size_t> result_shape(max_ndim, 1);

            for (const auto& input: inputs_)
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
            // return result_shape;
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
            if constexpr (Op::iter_kind() == IterationKind::ELEMENT_WISE) {
                return compute_strides_elementwise_();
            } else if constexpr (Op::iter_kind() == IterationKind::REDUCTION) {
                return compute_strides_reduction_<Op>();
            } else if constexpr (Op::iter_kind() == IterationKind::MATMUL) {
                return compute_strides_matmul_<Op>();
            } else if constexpr (Op::iter_kind() == IterationKind::SCALAR) {
                return compute_strides_scalar_();
            } else if constexpr (Op:iter_kind() == IterationKind::COPY) {
                return compute_strides_copy_();
            }
        }


        template <typename Op>
        std::vector<std::vector<size_t>> compute_strides_elementwise_()
        {
            if (Op::get_direction() == Direction::FORWARD) {

            }

            else if (Op::get_direction() == Direction::BACKWARD) { /* Placeholder */}
        }
        
        template <typename Op>
        std::vector<std::vector<size_t>> compute_strides_reduction_()
        {
            if (Op::get_direction() == Direction::FORWARD) {

            }

            else if (Op::get_direction() == Direction::BACKWARD) { /* Placeholder */}
        }
        
        template <typename Op>
        std::vector<std::vector<size_t>> compute_strides_matmul_()
        {
            if (Op::get_direction() == Direction::FORWARD) {

            }

            else if (Op::get_direction() == Direction::BACKWARD) { /* Placeholder */}
        }
        
        template <typename Op>
        std::vector<std::vector<size_t>> compute_strides_scalar_()
        {
            if (Op::get_direction() == Direction::FORWARD) {

            }

            else if (Op::get_direction() == Direction::BACKWARD) { /* Placeholder */}
        }
        
        template <typename Op>
        std::vector<std::vector<size_t>> compute_strides_copy_()
        {
            if (Op::get_direction() == Direction::FORWARD) {

            }

            else if (Op::get_direction() == Direction::BACKWARD) { /* Placeholder */}
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
                    inputs_[i] = inputs_[i]->to_dtype(common_dtype_);
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

        void add_input(const TensorImpl& tensor)
        {
            inputs_.push_back(std::make_shared<TensorImpl>(tensor));
        }
        
        void add_output(TensorImpl& tensor)
        {
            outputs_.push_back(std::make_shared<TensorImpl>(tensor));
        }
        
        /**
         * Checks correctenss by validating the inputs, computing the broadcasted shape and broadcasted strides
         * used by the kernels for accessing the tensor Storage elements.  
         */
        template <typename Op>
        void build(std::vector<size_t>& shape = {}, ScalarType cast_type = ScalarType::EMPTY)
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
                    output_shape_ = std::move(shape);
                
                } else {
                    broadcasted_shape_ = broadcast_shapes_<Op>();
                    output_shape_ = broadcasted_shape_;
                }

                if (outputs_.empty()) {
                    outputs_.resize(1);
                }

                if (!outputs_[0]) {
                    outputs_[0] = std::make_shared<TensorImpl>(output_shape_, common_dtype_, common_device_, common_requires_grad_);
                }

                if(common_requires_grad_) {
                    // Initialize grads: outputs_[0]->init_autograd_meta()
                    // Store input shapes in output_tensor->autograd_meta_->grad_fn_->input_shapes_;
                    //      required for backward pass
                }
            }
            
            else if (Op::get_direction() == Direction::BACKWARD) {

                if (inputs_.size() != Op::num_outputs()) {
                    throw std::runtime_error("Backward ERROR ....")
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

            broadcasted_strides_ = compute_broadcast_strides_<Op>();
            common_is_contiguous_ = check_contiguous();
        }

        
        std::vector<std::shared_ptr<TensorImpl>>& get_outputs()
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
        const void* output_data() const;
        
        void* output_grad_data();
        const void* output_grad_data() const;
        
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