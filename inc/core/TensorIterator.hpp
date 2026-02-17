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

    enum class IterationKind { ELEMENT_WISE, REDUCTION, MATMUL, SCALAR, COPY };
    enum class Direction { FORWARD, BACKWARD };

    class TensorIterator {
    private: 
        std::vector<std::shared_ptr<TensorImpl>> inputs_;
        std::shared_ptr<TensorImpl> output_;

        ScalarType common_dtype_;
        Device common_device_;
        bool is_contiguous_ = false;
        bool requires_grad_ = false;

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
            if (inputs_.empty() && !output_) {
                return ScalarType::Float32;
            }

            ScalarType promoted = ScalarType::Bool;
            
            if (output_) {
                promoted = output_->get_dtype();
            }

            for (const auto& input : inputs_)
            {
                if(!input) continue;
                promoted = promote_types(promoted, input->get_dtype());
            }
            
            promoted = promote_types(promoted, output_->get_dtype());

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
            
            if (output_) {
                common_dev = output_->get_device();
                device_initialized = true;
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
            if (output_) return output_->requires_grad();

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
            requires_grad_ = compute_requires_grad_();
        }

        
        // -------------------------------------------------------------------------------------------------------------  
        //                                                SHAPES BROADCASTING
        // ------------------------------------------------------------------------------------------------------------- 


        template <typename Op>
        std::vector<size_t> broadcast_shapes_()
        {
            if constexpr (Op::iter_kind() == IterationKind::ELEMENT_WISE) {
                if (inputs_.empty()) {
                    return output_ ? output_->get_shape() : throw std::runtime_error("NOT SURE CHECK AGAIN")
                } else {
                    return broadcast_shapes_elementwise_();
                }
                return broadcast_shapes_elementwise();
            } else if constexpr (Op::iter_kind() == IterationKind::REDUCTION) {
                return broadcast_shapes_reduction<Op>();
            } else if constexpr (Op::iter_kind() == IterationKind::MATMUL) {
                return broadcast_shapes_matmul_<Op>();
            } if constexpr (Op::iter_kind() == IterationKind::SCALAR) {
                return broadcast_shapes_scalar_();
            } if constexpr (Op::iter_kind() == IterationKind::COPY) {
                
            }
        }

        /**
         * Defines the common resulting shape for all the input tensors. Provide different shape computation
         * paths based on the required operation types.
         */
        std::vector<size_t> broadcast_shapes_elementwise_()
        {
            // TODO: requires the use of the REDUCTION path and MATMUL

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

            // Validate the output shape
            if (output_) {
                const auto& output_shape = output_->get_shape();

                if (output_shape.size() != result_shape.size()) {
                    throw std::runtime_error("...Shape dimensionality mismatch...");
                }

                for (size_t i = 0; i < result_shape.size(); ++i) {
                    if (output_shape[i] != result_shape[i]) {
                        throw std::runtime_error("...Mismatch in one dimension...");
                    }
                }
            }
            return result_shape;
        }

        template <typename Op>
        std::vector<size_t> broadcast_shapes_reduction_();
        
        template <typename Op>
        std::vector<size_t> broadcast_shapes_matmul_();

        std::vector<size_t> broadcast_shapes_scalar_();


        // -------------------------------------------------------------------------------------------------------------  
        //                                                  STRIDES BROADCASTING
        // ------------------------------------------------------------------------------------------------------------- 

        template <typename Op>
        std::vector<std::vector<size_t>> compute_broadcast_strides_()
        {
            if constexpr (Op:iter_kind() == IterationKind::ELEMENT_WISE) {
                return compute_strides_elementwise_();
            } else if constexpr (Op:iter_kind() == IterationKind::REDUCTION) {
                return compute_strides_reduction_<Op>();
            } else if constexpr (Op:iter_kind() == IterationKind::MATMUL) {
                return compute_strides_matmul_<Op>();
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
            // ...
        }
        
        void add_output(TensorImpl& tensor)
        {
            output_ = std::make_shared<TensorImpl>(tensor);
        }
        
        template <typename Op>
        void build(std::vector<size_t>& shape = {})
        {
            validate_inputs_metadata_<Op>();

            broadcasted_shape_ = broadcast_shapes_<Op>();
            
            if (output_) {
                output_shape_ = output_->get_shape();
            } else {
                output_shape_ = broadcasted_shape_;
            }
            
            if (!output_) {
                output_ = std::make_shared<TensorImpl>(output_shape_, common_device_, common_dtype_, requires_grad_);
            }

            broadcasted_strides_ = compute_broadcast_strides_<Op>();
            is_contiguous_ = check_contiguous();
        }

        // Which one and why?
        std::shared_ptr<TensorImpl> get_outputt()
        {
            return output_;
        }

        TensorImpl& get_output()
        {
            if (!output_) {
                throw std::runtime_error("Output tensor not initialized");
            }

            return *output_;
        }

        // --------------------------- KERNEL ---------------------------

        void* input_data(int idx);
        const void* input_data(int idx) const;
        
        void* output_data();
        const void* output_data() const;

    };

} //namespace tensor