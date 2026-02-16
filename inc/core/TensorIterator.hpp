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

    class TensorIterator {
    private: 
        std::vector<std::shared_ptr<TensorImpl>> inputs_;
        std::shared_ptr<TensorImpl> output_;

        ScalarType common_dtype_;
        Device common_device_;
        bool is_contiguous_ = false;
        bool requires_grad_ = false;

        std::vector<size_t> output_shape_;
        std::vector<std::vector<size_t>> input_strides_;
        std::vector<std::vector<size_t>> broadcasting_strides_;

        // template <typename Op>
        // std::vector<size_t> compute_output_shape_()
        // {
        // }

        // template <typename Op>
        // void compute_broadcast_strides_();

        /**
         * Compute the output type using the type promotion rules and the ScalarType
         */
        ScalarType compute_common_dtype_(ScalarType preferred_type = ScalarType::Float32)
        {
            if (inputs_.empty() && !output_) {
                return preferred_type;
            }

            ScalarType promoted = inputs_[0]->get_dtype();

            for (size_t i = 1; i < inputs_.size(); ++i)
            {
                promoted = promote_types(promoted, inputs_[i]->get_dtype());
            }

            promoted = promote_types(promoted, preferred_type);

            return promoted;
        }

        Device compute_common_device_()
        {
            Device common_dev;
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
        
        bool check_contiguous_()
        {
            if (output_ && !output_->is_contiguous())
                return false;
            
            for (const auto& input : inputs_)
            {
                if(!input->is_contiguous() || input->get_shape() != output_shape_)
                    return false;
            }
            return true;
        }

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
         * Defines the common resulting shape for all the input tensors
         */
        std::vector<size_t> broadcast_shapes_()
        {
            // TODO: requires the use of the REDUCTION path and MATMUL

            if (inputs_.empty()) {
                return {};
            }

            size_t max_ndim = 0;
            for (const auto& input: inputs_) {
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

            return result_shape;
        }


        /**
         * Initialize the resulting shape vector with the size of the biggest tensor. vector: 1,
         * matrix:2, tensor:3, batch:4).
         * 
         * Use shape and dimension number from each tensor:
         *  
         * 
        */

        template <typename Op>
        void validate_inputs_();

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
        void build()
        {
            output_shape_ = broadcast_shapes_();
            common_device_ = compute_common_device_();
            common_dtype_ = compute_common_dtype_();

            if (!output_) {
                output_ = std::make_shared<TensorImpl>(output_shape_, common_device_, common_dtype_, requires_grad_);
            }

            is_contiguous_ = check_contiguous();

            if (Op::iter_kind_ == IterationKind::ELEMENT_WISE)
            {
                if (!is_contiguous_) {
                    compute_broadcast_strides_<Op>();
                }
            }
            // validate_inputs_<Op>();
        }

        TensorImpl get_output();

        // --------------------------- KERNEL ---------------------------

        void* input_data(int idx);
        const void* input_data(int idx) const;
        
        void* output_data();
        const void* output_data() const;

    };

} //namespace tensor