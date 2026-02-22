#include "core/TensorImpl.hpp"
#include "core/TensorIterator.hpp"
#include <algorithm>

/**
 * TODO: check which templates require explicit instantiations
 */
namespace tensor
{

    // -------------------------------------------------------------------------------------------------------------  
    //                                                METADATA VALIDATION
    // -------------------------------------------------------------------------------------------------------------

    ScalarType TensorIterator::compute_common_dtype_()
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


    Device TensorIterator::compute_common_device_()
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


    bool TensorIterator::check_contiguous_()
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

    bool TensorIterator::compute_requires_grad_() {

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


    template<typename Op>
    void TensorIterator::validate_inputs_metadata_()
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
       
    // TODO: SEPARATE broadcasting and validation (validation is simply a size vector comparison).
    // TODO-fix: The return types assumes always a single output. Might require more than one (they might be of the same
    // shape in any relevant case but not changing would be a bad approach) 
    template <typename Op>
    std::vector<std::vector<size_t>> TensorIterator::broadcast_shapes_()
    {
        if constexpr      (Op::iter_kind() == IterationKind::ELEMENT_WISE)  return broadcast_shapes_elementwise_<Op>();
        else if constexpr (Op::iter_kind() == IterationKind::REDUCTION)     return broadcast_shapes_reduction_<Op>();
        else if constexpr (Op::iter_kind() == IterationKind::MATMUL)        return broadcast_shapes_matmul_<Op>();
        else if constexpr (Op::iter_kind() == IterationKind::SCALAR)        return broadcast_shapes_scalar_<Op>();
        else if constexpr (Op::iter_kind() == IterationKind::COPY)          return broadcast_shapes_copy_<Op>();

        // TODO: if(outputs_): compare the computed output shapes to the previous one.
        // If shapes are not the same throw error (do not change the previous existing tensor shape).

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
    }

    template <typename Op>
    std::vector<std::vector<size_t>> TensorIterator::broadcast_shapes_elementwise_()
    {
        std::vector<std::vector<size_t>> computed_shapes;
        computed_shapes.resize(outputs_.size());

        // Forward operations havea single output so looping over outputs_ is not required
        if (Op::get_direction() == Direction::FORWARD) {

            // FAST PATH 1: single operand. Forward the shape to output.
            if(inputs_.size() == 1) {
                computed_shapes[0] = inputs_[0]->get_shape();
                return computed_shapes;

            } else {
                
                // FAST PATH 2: if all inputs have same shape, forward shape to output.
                std::vector<size_t> temp{};
                bool shapes_match = true;
                
                for (auto& input : inputs_) {
                    if(!temp.empty() && temp != input->get_shape()) {
                        shapes_match = false;
                        break;
                    }
                    temp = input->get_shape();
                }
                
                if (shapes_match)
                {
                    computed_shapes[0] = temp;
                    return computed_shapes;
                }
            }

            throw std::runtime_error("Acutal broadcasting is not implemented yet"); 

            /* IMPLEMENTATION OF THE GENERAL CASE (CONSIDER ONLY FORWARD 1 OUTPUT) */
            /**
             * Different sized shapes: check if it can be left padded with ones. 
             * There are probably some weird cases of empty tensors that will break this
             * ....
             * 
             * If one of the tensor has a 0 dimension this should throw. This has to be assessed here but where in the pipeline?
             * Probably at the beginning (common to forward and backward) but looping dimensions should also achieve other stuff in the mean time.
             * 
             * TODO: fix the following incomplete implementation:
             *       Should iterate from the rightmost value of each input shape, check for all inputs if the value is the same, different from 0
             *       or they are all 1s except for a single value.
             *       (This while considering a single output since it is the forward version). 
             */

            /*
            size_t max_ndim = 0;
            for (auto& input : inputs_) {
                max_ndim = std::max(max_ndim, input->get_shape().size());
            }


            for (size_t i = 0; i < padded_sizes.size(); ++i) {
                for (auto& input : inputs_) {
                    input->get_shape()[input->get_shape().size() - 1 - i];
                }
            }


            // Left padding with ones
            for (auto& input : inputs_) {
                
                auto input_shape = input->get_shape();
                auto input_shape_size = input_shape.size(); 
                size_t pad_offset = max_ndim - input_shape_size;
                
                if (pad_offset >= 1) {
                    std::vector<size_t> padded_size = input_shape.insert(input_shape.begin(), 1);
                } else {
                    padded_sizes.push_back(input->get_shape().size())
                }
            }
            */
        }

        else if (Op::get_direction() == Direction::BACKWARD) {
            throw std::runtime_error("backward elementwise shape broadcasting not implemented");
        }

        return computed_shapes;
    }  

    template <typename Op>
    std::vector<std::vector<size_t>> TensorIterator::broadcast_shapes_reduction_()
    {
        return {0}; // placeholder
    }
    
    template <typename Op>
    std::vector<std::vector<size_t>> TensorIterator::broadcast_shapes_matmul_()
    {
        return {0}; // placeholder
    }

    template <typename Op>
    std::vector<std::vector<size_t>> TensorIterator::broadcast_shapes_scalar_()
    {
        return {0}; // placeholder
    }
    
    template <typename Op>
    std::vector<std::vector<size_t>> TensorIterator::broadcast_shapes_copy_()
    {
        return {0}; // placeholder
    }


    // -------------------------------------------------------------------------------------------------------------  
    //                                                  STRIDES BROADCASTING
    // -------------------------------------------------------------------------------------------------------------

    template <typename Op>
    std::vector<std::vector<size_t>> TensorIterator::compute_broadcast_strides_()
    {
        if      constexpr (Op::iter_kind() == IterationKind::ELEMENT_WISE) return compute_strides_elementwise_<Op>();
        else if constexpr (Op::iter_kind() == IterationKind::REDUCTION)    return compute_strides_reduction_<Op>();
        else if constexpr (Op::iter_kind() == IterationKind::MATMUL)       return compute_strides_matmul_<Op>();
        else if constexpr (Op::iter_kind() == IterationKind::SCALAR)       return compute_strides_scalar_<Op>();
        else if constexpr (Op::iter_kind() == IterationKind::COPY)         return compute_strides_copy_<Op>();

        // TODO: if(outputs_) check strides compatiblity. No hard requirement as in the shape, but at least the 
        // shape length must be the same. 
        // If they are compatible the previous strides can be substituted.
    }

    /**
     * Strides can be however changed due to the broadcasting logic.
     * TODO: provide fast path for operations that do not require it: filling, same size pointwise, other ...
     * 
     * Not sure how the forward and backward should be different. Maybe for this case
     * (element wise operations) can be the same, but not for the other ones.
     */
    template <typename Op>
    std::vector<std::vector<size_t>> TensorIterator::compute_strides_elementwise_()
    {
        std::vector<std::vector<size_t>> strides;

        if constexpr (Op::get_direction() == Direction::FORWARD) {

        }

        else if (Op::get_direction() == Direction::BACKWARD) {
            throw std::runtime_error("TensorIterator: backward elementwise not implemented");
        }
    }

    template <typename Op>
    std::vector<std::vector<size_t>> TensorIterator::compute_strides_reduction_(){
        if (Op::get_direction() == Direction::FORWARD) {
            throw std::runtime_error("TensorIterator: compute_strides_reduction_() not implemented");
        }
        else if (Op::get_direction() == Direction::BACKWARD) {
            throw std::runtime_error("TensorIterator: compute_strides_reduction_() backward not implemented");
        }
    }
    
    template <typename Op>
    std::vector<std::vector<size_t>> TensorIterator::compute_strides_matmul_(){
        if (Op::get_direction() == Direction::FORWARD) {
            throw std::runtime_error("TensorIterator: compute_strides_matmul_() not implemented");
        }
        else if (Op::get_direction() == Direction::BACKWARD) {
            throw std::runtime_error("TensorIterator: compute_strides_matmul_() backward not implemented");
        }
    }
    
    template <typename Op>
    std::vector<std::vector<size_t>> TensorIterator::compute_strides_scalar_(){
        if (Op::get_direction() == Direction::FORWARD) {
            throw std::runtime_error("TensorIterator: compute_strides_scalar_() not implemented");
        }
        else if (Op::get_direction() == Direction::BACKWARD) {
            throw std::runtime_error("TensorIterator: compute_strides_scalar_() backward not implemented");
        }
    }
    
    template <typename Op>
    std::vector<std::vector<size_t>> TensorIterator::compute_strides_copy_(){
        if (Op::get_direction() == Direction::FORWARD) {
            throw std::runtime_error("TensorIterator: compute_strides_copy_() not implemented");
        }
        else if (Op::get_direction() == Direction::BACKWARD) {
            throw std::runtime_error("TensorIterator: compute_strides_copy_() backward not implemented");
        }
    }


    // -------------------------------------------------------------------------------------------------------------  
    //                                                  TYPES MATERIALIZATION
    // ------------------------------------------------------------------------------------------------------------- 

    void TensorIterator::materialize_inputs_()
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


    TensorIterator::TensorIterator() = default;
    TensorIterator::TensorIterator(const TensorIterator&) = delete;
    TensorIterator& TensorIterator::operator=(const TensorIterator&) = delete;
    TensorIterator::TensorIterator(TensorIterator&&) = default;
    TensorIterator& TensorIterator::operator=(TensorIterator&&) = default;

    bool TensorIterator::get_inplace()         { return inplace_; }
    void TensorIterator::set_inplace(bool val) { inplace_ = val; }


    // -------------------------------------------------------------------------------------------------------------  
    //                                                DISPATCHER INTERFACES
    // ------------------------------------------------------------------------------------------------------------- 

    void TensorIterator::add_input(TensorImpl* tensor)
    {
        inputs_.push_back(tensor);
    }
    
    void TensorIterator::add_output(TensorImpl* tensor)
    {
        outputs_.push_back(tensor);
    } 

    template <typename Op>
    void TensorIterator::build(const ScalarType cast_type = ScalarType::EMPTY)
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
            // TODO: add check also on the operation template (if std::is_base_of_v<CastOp, Op>)
            if (cast_type != ScalarType::EMPTY) {
                if(shape.empty()) {
                    throw std::runtime_error("Cannot cast an empty tensor");
                }
                common_dtype_ = cast_type;
            }
            
            broadcasted_shapes_ = broadcast_shapes_<Op>();
            output_shapes_ = broadcasted_shapes_;
            
            if (outputs_.empty()) {
                outputs_.push_back(nullptr);
            }

            if (!outputs_[0]) {
                nullary_output_ = std::make_unique<TensorImpl>(output_shapes_[0], common_dtype_, common_device_, common_requires_grad_);
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

    
    std::vector<TensorImpl*> TensorIterator::get_outputs()
    {
        return outputs_;
    }


    // -------------------------------------------------------------------------------------------------------------  
    //                                                KERNEL INTERFACES
    // -------------------------------------------------------------------------------------------------------------

    void* TensorIterator::input_data(int idx)
    {
        if (idx < 0 || static_cast<size_t>(idx) >= inputs_.size())
            throw std::out_of_range("input_data: out_of range");
        return inputs_[idx]->data_ptr();
    }

    const void* TensorIterator::input_data(int idx) const
    {
        // placeholder
    }
    
    void* TensorIterator::output_data(int idx)
    {
        if (idx < 0 || idx >= static_cast<int>(outputs_.size()))
        throw std::out_of_range("output_data: index out of range");
        return outputs_[idx]->data_ptr();
    }

    const void* TensorIterator::output_data(int idx) const
    {
        // placeholder
    }
    
    void* TensorIterator::output_grad_data(int idx)
    {
        return nullptr; // placeholder
    }

    const void* TensorIterator::output_grad_data(int idx) const
    {
        return nullptr; // placeholder
    }
    

    template <typename T>
    T* TensorIterator::input_ptr(int idx)
    {
        return static_cast<T*>(input_data(idx));
    }

    template <typename T>
    T* TensorIterator::output_ptr(int idx = 0)
    {
        return static_cast<T*>(output_data(idx));
    }

} //namespace tensor