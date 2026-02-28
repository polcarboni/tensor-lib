#include "core/TensorImpl.hpp"
#include "core/TensorIterator.hpp"
#include "ops/OpsRegistry.hpp"
#include <algorithm>

/**
 * TODO: check which templates require explicit instantiations
 */
namespace tensor
{

    
    //               ##   ##  ####### ########  #####  ########     #####  ########  #####  
    //               ### ###  ##         ##    ##   ## ##     ##   ##   ##    ##    ##   ## 
    //               ## # ##  ##         ##    ##   ## ##     ##   ##   ##    ##    ##   ## 
    //               ##   ##  ######     ##    ####### ##     ##   #######    ##    ####### 
    //               ##   ##  ##         ##    ##   ## ##     ##   ##   ##    ##    ##   ## 
    //               ##   ##  ##         ##    ##   ## ##     ##   ##   ##    ##    ##   ## 
    //               ##   ##  #######    ##    ##   ## ########    ##   ##    ##    ##   ## 


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

        common_device_        = compute_common_device_();
        common_dtype_         = compute_common_dtype_();
        common_requires_grad_ = compute_requires_grad_();
    }

        



    //                                ######   ##     ##    ###    ########  ########  ######  
    //                               ##    ##  ##     ##   ## ##   ##     ## ##       ##    ## 
    //                               ##        ##     ##  ##   ##  ##     ## ##       ##       
    //                                ######   ######### ##     ## ########  ######    ######  
    //                                     ##  ##     ## ######### ##        ##             ## 
    //                               ##    ##  ##     ## ##     ## ##        ##       ##    ## 
    //                                ######   ##     ## ##     ## ##        ########  ######


    // -------------------------------------------------------------------------------------------------------------  
    //                                                SHAPES BROADCASTING
    // ------------------------------------------------------------------------------------------------------------- 

    /**
     * Checks the correct use of inline operations by validating the shape computed by the broadcasting
     * again the provided output shape tensors.
     */
    void validate_output_shapes_(std::vector<std::vector<size_t>>& output_shapes,
                                 std::vector<std::vector<size_t>>& computed_shapes)
    {
        if (computed_shapes.size() != output_shapes.size()) {
            throw std::runtime_error("TensorIterator: number of computed output shapes (" +
                                        std::to_string(computed_shapes.size()) +
                                        ") does not match number of outputs (" +
                                        std::to_string(output_shapes.size()) + ")");
        }
        

        for (size_t i = 0; i < output_shapes.size(); ++i) {

            const auto& output_shape = output_shapes[i];
            const auto& computed_shape = computed_shapes[i];

            if (output_shape.size() != computed_shape.size()) {
                throw std::runtime_error("TensorIterator: output[" + std::to_string(i) +
                                         "] dimensionality mismatch: existing rank=" +
                                         std::to_string(output_shape.size()) +
                                         ", computed rank=" + std::to_string(computed_shape.size()));
            }

            for (size_t dim = 0; dim < computed_shape.size(); ++dim) {
                if(output_shape[dim] != computed_shape[dim]) {
                    throw std::runtime_error("TensorIterator: output[" + std::to_string(i) +
                                             "] shape mismatch at dim " + std::to_string(dim) +
                                             ": existing=" + std::to_string(output_shape[dim]) +
                                             ", computed=" + std::to_string(computed_shape[dim]));
                }
            }
        }
    }

       
    // TODO: SEPARATE broadcasting and validation (validation is simply a size vector comparison).
    // TODO-fix: The return types assumes always a single output. Might require more than one (they might be of the same
    // shape in any relevant case but not changing would be a bad approach) 
    template <typename Op>
    std::vector<std::vector<size_t>> TensorIterator::broadcast_shapes_()
    {
        std::vector<std::vector<size_t>> computed_shapes;

        if constexpr      (Op::iter_kind() == IterationKind::ELEMENT_WISE)  computed_shapes = broadcast_shapes_elementwise_<Op>();
        else if constexpr (Op::iter_kind() == IterationKind::REDUCTION)     computed_shapes = broadcast_shapes_reduction_<Op>();
        else if constexpr (Op::iter_kind() == IterationKind::MATMUL)        computed_shapes = broadcast_shapes_matmul_<Op>();
        else if constexpr (Op::iter_kind() == IterationKind::COPY)          check_shapes_copy_<Op>();

        
        // =================================== OUTPUT SHAPES VALIDATION ===================================
        // TODO: also check the inplace_ data member? remove the data member?
        
        if constexpr (Op::iter_kind() != IterationKind::COPY) {
            if (!outputs_.empty()) {
                validate_output_shapes_(output_shapes_, computed_shapes);
            }
        }

        return computed_shapes;
    }

    template <typename Op>
    std::vector<std::vector<size_t>> TensorIterator::broadcast_shapes_elementwise_()
    {
        std::vector<std::vector<size_t>> computed_shapes;
        computed_shapes.resize(Op::num_outputs());
        
        is_broadcasted_ = false;

        // Forward operations havea single output so looping over outputs_ is not required
        
        // ============================ FORWARD BROADCASTING ============================

        if (Op::get_direction() == Direction::FORWARD) {

            // ------------------------ SCALAR OPERATIONS ------------------------
            if (scalar_) {
                is_broadcasted_ = true;
                computed_shapes[0] = inputs_[0]->get_shape();
                return computed_shapes;
            }

            // Check if input tensors are empty (throw if an input operand is empty)
            for (const auto& input : inputs_) {
                const auto& shape = input->get_shape();
                for (auto dim : shape) {
                    if (dim == 0) {
                        throw std::runtime_error("broadcast_shapes_elementwise: provided empty tensor");
                    }
                }
            }

            // ---------------------- fast path 1: single operand ----------------------
            
            if (inputs_.size() == 1) {
                computed_shapes[0] = inputs_[0]->get_shape();
                return computed_shapes;
                
            }
            
            // ---------------------- fast path 2: same shapes ----------------------
            const auto& first_shape = inputs_[0]->get_shape();
            for (size_t i = 1; i < inputs_.size(); ++i) {
                if (inputs_[i]->get_shape() != first_shape) {
                    is_broadcasted_ = true;
                    break;
                }
            }
            
            if (!is_broadcasted_)
            {
                computed_shapes[0] = first_shape;
                return computed_shapes;
            }
                        
            // ------------------------- broadcasting logic -------------------------
            
            size_t max_rank = 0;
            for (auto& input : inputs_) {
                max_rank = std::max(max_rank, input->get_shape().size());
            }
            
            // Initialize the output shape with 1s
            std::vector<size_t> output_shape(max_rank, 1);
            
            // Iterate over inputs dimensions to update output_shape
            for (auto& input : inputs_) {
                
                auto input_shape = input->get_shape();
                auto input_rank = input_shape.size(); 
                size_t pad_offset = max_rank - input_rank;
                
                // Compare/accumulate input dimensions with broadcasting rules
                for (size_t i = 0; i < input_rank; ++i) {
                    size_t input_dim = input_shape[i];
                    size_t& output_dim = output_shape[i + pad_offset];
                    
                    if (input_dim == 1) {
                        continue;
                    } else if (output_dim == 1) {
                        output_dim = input_dim;
                    } else if (output_dim != input_dim) {
                        throw std::runtime_error("Incompatible shapes for broadcasting");
                    }
                }
            }

            computed_shapes[0] = output_shape;
        }

        // ============================ BACKWARD BROADCASTING ============================

        else if (Op::get_direction() == Direction::BACKWARD) {
            throw std::runtime_error("backward elementwise shape broadcasting not implemented");
        }

        return computed_shapes;
    }  



    template <typename Op>
    std::vector<std::vector<size_t>> TensorIterator::broadcast_shapes_reduction_()
    {
        std::vector<std::vector<size_t>> computed_shapes;
        computed_shapes.resize(outputs_.size());

        // ============================ FORWARD BROADCASTING ============================

        if (Op::get_direction() == Direction::FORWARD) {

            if(inputs_.size() != 1) {
                throw std::runtime_error("TensorIterator: reduction operations must have 1 input");
            }

            auto input       = inputs_[0];
            auto input_shape = input->get_shape();
            auto input_rank  = input_shape.size();


            if (!reduction_axes_) {
                computed_shapes[0] = input_shape;
                return computed_shapes;
            }

            auto& r_axes = *reduction_axes_;

            // Full reduction (no axes provided)
            if (r_axes.empty()) {
                computed_shapes[0] = std::vector<size_t>{1};
                return computed_shapes;
            }

            for (auto ax : r_axes) {
                if (ax >= input_rank) {
                    throw std::runtime_error("Reduction: the provided dimensions are not correct");
                } 
            }


            std::vector<size_t> output_shape;

            for (size_t i = 0; i < input_rank; ++i) {
                if (std::find(r_axes.cbegin(), r_axes.cend(), i) != r_axes.cend()) {
                    if (keepdims_) {
                        output_shape.push_back(1);
                    }  // else: reduced axis, skip. 
                } else {
                        output_shape.push_back(input_shape[i]);
                }
            }
            computed_shapes[0] = output_shape;
        }

        // ============================ BACKWARD BROADCASTING ============================
        
        else if (Op::get_direction() == Direction::BACKWARD) {
            throw std::runtime_error("Backward reduction: shape broadcasting not implemented");
        }

        return computed_shapes;
    }
    
    template <typename Op>
    std::vector<std::vector<size_t>> TensorIterator::broadcast_shapes_matmul_()
    {
        std::vector<std::vector<size_t>> computed_shapes;
        computed_shapes.resize(outputs_.size());


        // ============================ FORWARD BROADCASTING ============================

        if (Op::get_direction() == Direction::FORWARD) {

            if (inputs_.size() != 2) {
                throw std::runtime_error("TensorIterator: Matmul must have 2 operands");
            }

            auto lhs_shape = inputs_[0]->get_shape();
            auto rhs_shape = inputs_[1]->get_shape();
            auto lhs_rank = lhs_shape.size();
            auto rhs_rank = rhs_shape.size();

            if (lhs_rank > 3 || rhs_rank > 3) {
                throw std::runtime_error("Matmul: inputs must be at most 3D");
            }

            if (lhs_rank == 3 && rhs_rank == 3 && lhs_shape[0] != rhs_shape[0] && lhs_shape[0] != 1 && rhs_shape[0] != 1) {
                throw std::runtime_error("Matmul: batch dimensions are incompatible");
            }

            // Output matrix shape
            size_t lhs_rows = lhs_shape[lhs_rank - 2];
            size_t lhs_cols = lhs_shape[lhs_rank - 1];
            size_t rhs_rows = rhs_shape[rhs_rank - 2];
            size_t rhs_cols = rhs_shape[rhs_rank - 1];

            if (lhs_cols != rhs_rows) {
                throw std::runtime_error("Matmul: inner dimensions do not match (" +
                                         std::to_string(lhs_cols) + " vs " +
                                         std::to_string(rhs_rows) + ")");
            }

            std::vector<size_t> output_shape;

            // Output batch size
            if (lhs_rank == 3 || rhs_rank == 3) {
                size_t lhs_batch = (lhs_rank == 3) ? lhs_shape[0] : 1;
                size_t rhs_batch = (rhs_rank == 3) ? rhs_shape[0] : 1;
                
                auto batch = std::max(lhs_batch, rhs_batch);
                output_shape.push_back(batch);
            }

            output_shape.push_back(lhs_rows);
            output_shape.push_back(rhs_cols);

            computed_shapes[0] = output_shape;
        }


        // ============================ BACKWARD BROADCASTING ============================

        else if (Op::get_direction() == Direction::BACKWARD) {
            throw std::runtime_error("Backward Matmul: shape broadcasting not implemented");
        }

        return computed_shapes;
    }
    
    template <typename Op>
    void TensorIterator::check_shapes_copy_()
    {

        // ============================ FORWARD BROADCASTING ============================

        if (Op::get_direction() == Direction::FORWARD) {

            if (inputs_.size() != 1) {
                throw std::runtime_error("TensorIterator: Copy operation must have 1 input");
            }

            auto input = inputs_[0];
            auto input_size = input->get_total_size();

            for (auto& output : outputs_) {
                if(output->get_total_size() != input_size) {
                    throw std::runtime_error("Copy operations: mismatch in input and output sizes");
                }
            
            }
        }

        // ============================ BACKWARD BROADCASTING ============================

        else if (Op::get_direction() == Direction::BACKWARD) {
            throw std::runtime_error("Backward Copy: shape broadcasting not implemented");
        }
    }







    //                                ######   ########  ########   ##  ########  ########  ######  
    //                               ##    ##     ##     ##     ##  ##  ##     ## ##       ##    ## 
    //                               ##           ##     ##     ##  ##  ##     ## ##       ##       
    //                                ######      ##     ########   ##  ##     ## ######    ######  
    //                                     ##     ##     ##   ##    ##  ##     ## ##             ## 
    //                               ##    ##     ##     ##    ##   ##  ##     ## ##       ##    ## 
    //                                ######      ##     ##     ##  ##  ########  ########  ######

    // -------------------------------------------------------------------------------------------------------------  
    //                                                  STRIDES BROADCASTING
    // -------------------------------------------------------------------------------------------------------------

    template <typename Op>
    std::vector<std::vector<size_t>> TensorIterator::compute_broadcast_strides_()
    {
        std::vector<std::vector<size_t>> computed_strides;

        if      constexpr (Op::iter_kind() == IterationKind::ELEMENT_WISE) computed_strides = compute_strides_elementwise_<Op>();
        else if constexpr (Op::iter_kind() == IterationKind::REDUCTION)    computed_strides = compute_strides_reduction_<Op>();
        else if constexpr (Op::iter_kind() == IterationKind::MATMUL)       computed_strides = compute_strides_matmul_<Op>();
        else if constexpr (Op::iter_kind() == IterationKind::COPY)         computed_strides = compute_strides_copy_<Op>();


        // =================================== OUTPUT STRIDES VALIDATION ===================================
        
        if (!outputs_.empty()) {
            const size_t offset = inputs_.size();

            for (size_t i = 0; i < outputs_.size(); ++i) {
                if (outputs_[i] == nullptr) continue;

                const size_t stride_idx = offset + i;
                if (stride_idx >= computed_strides.size()) continue;

                const std::vector<size_t>& computed_stride = computed_strides[stride_idx];
                const std::vector<size_t>& output_strides =  outputs_[i]->get_strides();

                // Hard check: computed strides size must match the output shape size
                if (computed_stride.size() != output_strides.size()) {
                    throw std::runtime_error("TensorIterator: output[" + std::to_string(i) +
                                             "] stride rank (" + std::to_string(computed_stride.size()) +
                                             ") does not match shape rank (" +
                                             std::to_string(output_strides.size()) + ")");
                }
                
                // Soft check: control if computed strides are different from original output strides
                // TODO: change the name of the variable (computed_stride is the computed stride for the output) 
                // but is too confusing
                for (size_t dim = 0; dim < computed_stride.size(); ++dim) {
                    if (output_strides[dim] != computed_stride[dim]) {
                        std::cerr << "[TensorIterator] warning: output[" << i
                                    << "] stride mismatch at dim " << dim
                                    << ": existing=" << output_strides[dim]
                                    << ", computed=" << computed_stride[dim]
                                    << " — substituting computed strides\n";
                        break;   
                    }
                }
                outputs_[i]->set_strides(computed_stride);
            }
        }

        return computed_strides;
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
        /**
         * TODOFIX: this currently does not use the input tensors strides data member.
         *          these should be directly forwarded to the output if there is no
         *          broadcasting involved (same shapes). 
         * 
         *          Not 100% sure. Does this function manage noncontiguous? 
         */

        std::vector<std::vector<size_t>> all_strides;
        all_strides.reserve(inputs_.size() + outputs_.size());

        // ============================ FORWARD BROADCASTING ============================

        if constexpr (Op::get_direction() == Direction::FORWARD) {
            
            // ---------------------- SCALAR OPERATIONS ------------------------
            if (scalar_) {
                for (const auto& input : inputs_) {
                    if (input->get_shape().empty()) {
                        all_strides.push_back({0});
                    } else {
                        all_strides.push_back(input->get_strides());
                    }
                }

                for (size_t i = 0; i < outputs_.size(); ++i) {
                    if (outputs_[i]->get_shape().empty()) {
                        all_strides.push_back({0});
                    } else {
                        all_strides.push_back(outputs_[i]->get_strides());
                    }
                }
                return all_strides;
            }

            const std::vector<size_t>& target_shape = output_shapes_[0];
            size_t max_rank = target_shape.size();


            // ---------------------- input strides ------------------------
            
            for (const auto& input : inputs_) {

                const std::vector<size_t>& actual_shape = input->get_shape();
                const std::vector<size_t>& actual_strides = input->get_strides();

                size_t input_rank = actual_shape.size();
                size_t pad_offset = max_rank - input_rank;
                
                std::vector<size_t> broadcasted_strides(max_rank, 0);
                
                for (size_t i = 0; i < input_rank; ++i) {
                    if (actual_shape[i] != 1) {
                        broadcasted_strides[i + pad_offset] = actual_strides[i];
                    } else {
                        broadcasted_strides[i + pad_offset] = 0;
                    }
                }

                all_strides.push_back(std::move(broadcasted_strides));
            }
            

            // ---------------------- output strides ------------------------
            
            for (size_t i = 0; i < outputs_.size(); ++i) {
                std::vector<size_t> out_strides(max_rank);
                size_t current_stride = 1;

                for (int d = static_cast<int>(max_rank) - 1; d >= 0; --d) {
                    out_strides[d] = current_stride;
                    current_stride *= target_shape[d];
                }
                
                all_strides.push_back(std::move(out_strides));
            }
        }

        // ============================ BACKWARD BROADCASTING ============================

        else if constexpr (Op::get_direction() == Direction::BACKWARD) {
            throw std::runtime_error("TensorIterator: backward elementwise not implemented");
        }

        return all_strides;
    }


    template <typename Op>
    std::vector<std::vector<size_t>> TensorIterator::compute_strides_reduction_()
    {
        std::vector<std::vector<size_t>> result;

        if (Op::get_direction() == Direction::FORWARD) {
            
            // Identification of input reduced axis 
            auto input = inputs_[0]; 
            auto input_strides = input->get_strides(); 
            auto ndim = input_strides.size(); 

            std::vector<bool> is_reduced(ndim, false);
            
            if (reduction_axes_.has_value()) {
                for (size_t ax : reduction_axes_.value()) {
                    is_reduced[ax] = true;
                }
            }

            // Does not use the strides data member for reasons I am not really familiar with:
            // permutation of dimensions, coalescing of dimensions (ndim smaller than the original one)

            // for now this could only be result.push_back(ipnut_strides);
            // I will leave this comment to remember to add the required improvements for performance.
            std::vector<size_t> input_iter_strides;
            for (size_t i = 0; i < ndim; ++i) {
                input_iter_strides.push_back(input_strides[i]);
            }
            result.push_back(std::move(input_iter_strides));


            // This also assumes single output. Some reduction opes could be different
            // but they might also use a separate function.

            auto output = outputs_[0]; 
            const auto& output_strides = output->get_strides(); 
            // auto output_strides_size = output_strides.size(); 
            std::vector<size_t> output_iter_strides;
            size_t output_axis_idx = 0;

            for (size_t i = 0; i < ndim; ++i) {
                if (is_reduced[i]) {
                    output_iter_strides.push_back(0);
                    if (keepdims_) {
                        output_axis_idx++;
                    }
                } else {
                    output_iter_strides.push_back(output_strides[output_axis_idx]);
                    output_axis_idx++;
                }
            }

            result.push_back(std::move(output_iter_strides));
        }

        else if (Op::get_direction() == Direction::BACKWARD) {
            throw std::runtime_error("TensorIterator: compute_strides_reduction_() backward not implemented");
        }

        return result;
    }
    
    template <typename Op>
    std::vector<std::vector<size_t>> TensorIterator::compute_strides_matmul_()
    {
        if (Op::get_direction() == Direction::FORWARD) {
            throw std::runtime_error("TensorIterator: compute_strides_matmul_() not implemented");
        }
        else if (Op::get_direction() == Direction::BACKWARD) {
            throw std::runtime_error("TensorIterator: compute_strides_matmul_() backward not implemented");
        }
    }
    
    template <typename Op>
    std::vector<std::vector<size_t>> TensorIterator::compute_strides_copy_()
    {
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






    //          ######    #######     ###    ##       ########  ######   ######  #### ##    ##  ######   
    //         ##    ##  ##     ##   ## ##   ##       ##       ##    ## ##    ##  ##  ###   ## ##    ##  
    //         ##        ##     ##  ##   ##  ##       ##       ##       ##        ##  ####  ## ##        
    //         ##        ##     ## ##     ## ##       ######    ######  ##        ##  ## ## ## ##   ####  
    //         ##        ##     ## ######### ##       ##             ## ##        ##  ##  #### ##    ##  
    //         ##    ##  ##     ## ##     ## ##       ##       ##    ## ##    ##  ##  ##   ### ##    ##  
    //          ######    #######  ##     ## ######## ########  ######   ######  #### ##    ##  ######
    

    // -------------------------------------------------------------------------------------------------------------  
    //                                                  DIMENSIONS COALESCING
    // ------------------------------------------------------------------------------------------------------------- 

    template <typename Op>
    std::vector<bool> TensorIterator::compute_merge_decision_(std::vector<std::vector<size_t>>& shapes,
                                                              std::vector<std::vector<size_t>>& strides)
    {
        std::vector<bool> merge_decision;
        
        if      constexpr (Op::iter_kind() == IterationKind::ELEMENT_WISE) merge_decision = compute_merge_decision_elementwise_<Op>(shapes, strides);
        else if constexpr (Op::iter_kind() == IterationKind::REDUCTION)    merge_decision = compute_merge_decision_reduction_<Op>(shapes, strides);
        else if constexpr (Op::iter_kind() == IterationKind::MATMUL)       merge_decision = compute_merge_decision_matmul_<Op>(shapes, strides);
        else if constexpr (Op::iter_kind() == IterationKind::COPY)         merge_decision = compute_merge_decision_copy_<Op>(shapes, strides);
        
        return merge_decision;
    }

    template <typename Op>
    std::vector<bool> TensorIterator::compute_merge_decision_elementwise_(std::vector<std::vector<size_t>>& shapes,
                                                                        std::vector<std::vector<size_t>>& strides)
    {
        return {}; // placeholder
    }

    template <typename Op>
    std::vector<bool> TensorIterator::compute_merge_decision_reduction_(std::vector<std::vector<size_t>>& shapes,
                                                                        std::vector<std::vector<size_t>>& strides)
    {
        return {}; // placeholder
    }

    template <typename Op>
    std::vector<bool> TensorIterator::compute_merge_decision_matmul_(std::vector<std::vector<size_t>>& shapes,
                                                                    std::vector<std::vector<size_t>>& strides)
    {
        return {}; // placeholder
    }



    template <typename Op>
    std::vector<bool> TensorIterator::compute_merge_decision_copy_(std::vector<std::vector<size_t>>& shapes,
                                                                std::vector<std::vector<size_t>>& strides)
    {
        return {}; // placeholder
    }

    std::vector<size_t> TensorIterator::apply_merge_to_shape_(std::vector<size_t>& shape,
                                                              std::vector<bool>& merge_decision)
    {
        return std::vector<size_t>(); // placeholder
    }

    std::vector<std::vector<size_t>> TensorIterator::apply_merge_to_strides_(std::vector<std::vector<size_t>>& strides,
                                                                             std::vector<bool>& merge_decision)
    {
        return std::vector<std::vector<size_t>>(); // placeholder
    }

    // TODO-fix: consider providing also coalesced_shape_ and coalesced_strides_ also as 
    // explicit args, not only as data members modified by the function.
    template <typename Op>
    bool TensorIterator::coalesce_dimensions_(std::vector<std::vector<size_t>>& shapes,
                                              std::vector<std::vector<size_t>>& strides)
    {
        try
        {
            std::vector<bool> merge_decision = compute_merge_decision_<Op>(shapes,strides);
            coalesced_shape_   = apply_merge_to_shape_  (shapes[0], merge_decision);
            coalesced_strides_ = apply_merge_to_strides_(strides,   merge_decision);
            return true;
        } catch (...)
        {
            // Coalescing is not possible
            return false;
        }
    }









    //      ########  ##    ##  ########  ########  ########  ########    ###     ######   ########  ######  
    //         ##     ###   ##     ##     ##        ##     ## ##         ## ##   ##    ##  ##       ##    ## 
    //         ##     ####  ##     ##     ##        ##     ## ##        ##   ##  ##        ##       ##       
    //         ##     ## ## ##     ##     ######    ########  ######   ##     ## ##        ######    ######  
    //         ##     ##  ####     ##     ##        ##   ##   ##       ######### ##        ##             ## 
    //         ##     ##   ###     ##     ##        ##    ##  ##       ##     ## ##    ##  ##       ##    ## 
    //      ########  ##    ##     ##     ########  ##     ## ##       ##     ##  ######   ########  ######
    



    // =============================================================================================================  
    // =============================================================================================================  
    //                                                PUBLIC INTERFACES
    // =============================================================================================================
    // =============================================================================================================

    // -------------------------------------------------------------------------------------------------------------  
    //                                                      GETTERS
    // ------------------------------------------------------------------------------------------------------------- 
    
    std::vector<TensorImpl*> TensorIterator::get_outputs() { return outputs_; }
    
    bool TensorIterator::get_inplace()              { return inplace_; }
    ScalarType TensorIterator::get_common_dtype()   { return common_dtype_; }
    Device TensorIterator::get_common_device()      { return common_device_; }
    bool TensorIterator::get_common_is_contiguous() { return common_is_contiguous_; }
    bool TensorIterator::get_common_requires_grad() { return common_requires_grad_; }
    bool TensorIterator::get_scalar()               { return scalar_; }
    
    void TensorIterator::set_inplace(bool val) { inplace_ = val; }
    void TensorIterator::set_reduction_axes(std::optional<std::vector<size_t>> axes) { reduction_axes_ = std::move(axes); }
    void TensorIterator::set_keepdims(bool keepdims) { keepdims_ = keepdims; }
    void TensorIterator::set_scalar(bool scalar)     { scalar_ = scalar; }









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
        output_shapes_.push_back(tensor->get_shape());
    } 

    template <typename Op>
    void TensorIterator::build(const ScalarType cast_type)
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

                if (inputs_.size() > 1) {
                    throw std::runtime_error("build(): casting operations only support 1 input");
                }

                if(inputs_[0]->get_shape().empty()) {
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

        // CHECK BEFORE ACTUALLY USING IT
        // if(!common_is_contiguous_) {
        //     coalesce_dimensions_(broadcasted_shapes_, broadcasted_strides_);
        // }
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
        return nullptr; // placeholder
    }
    
    void* TensorIterator::output_data(int idx)
    {
        if (idx < 0 || idx >= static_cast<int>(outputs_.size()))
        throw std::out_of_range("output_data: index out of range");
        return outputs_[idx]->data_ptr();
    }

    const void* TensorIterator::output_data(int idx) const
    {
        return nullptr; // placeholder
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
    T* TensorIterator::output_ptr(int idx)
    {
        return static_cast<T*>(output_data(idx));
    }

    







    /* Explicit instantiations */

    #define INSTANTIATE_T(T)                            \
        template T* TensorIterator::input_ptr<T>(int);  \
        template T* TensorIterator::output_ptr<T>(int);

    INSTANTIATE_T(float)
    INSTANTIATE_T(double)
    INSTANTIATE_T(int32_t)
    INSTANTIATE_T(int64_t)
    INSTANTIATE_T(bool)

    #define INSTANTIATE_OP(Op)                                                      \
        template void TensorIterator::build<::tensor::ops::Op>(                     \
            const ScalarType                                                        \
        );                                                                          \
        template std::vector<bool>                                                  \
        TensorIterator::compute_merge_decision_<::tensor::ops::Op>(                 \
            std::vector<std::vector<size_t>>&,                                      \
            std::vector<std::vector<size_t>>&                                       \
        );                                                                          \
        template std::vector<bool>                                                  \
        TensorIterator::compute_merge_decision_elementwise_<::tensor::ops::Op>(     \
            std::vector<std::vector<size_t>>&,                                      \
            std::vector<std::vector<size_t>>&                                       \
        );                                                                          \
        template std::vector<bool>                                                  \
        TensorIterator::compute_merge_decision_reduction_<::tensor::ops::Op>(       \
            std::vector<std::vector<size_t>>&,                                      \
            std::vector<std::vector<size_t>>&                                       \
        );                                                                          \
        template std::vector<bool>                                                  \
        TensorIterator::compute_merge_decision_matmul_<::tensor::ops::Op>(          \
            std::vector<std::vector<size_t>>&,                                      \
            std::vector<std::vector<size_t>>&                                       \
        );                                                                          \
        template std::vector<bool>                                                  \
        TensorIterator::compute_merge_decision_copy_<::tensor::ops::Op>(            \
            std::vector<std::vector<size_t>>&,                                      \
            std::vector<std::vector<size_t>>&                                       \
        );                                                                          \
        template bool                                                               \
        TensorIterator::coalesce_dimensions_<::tensor::ops::Op>(                    \
            std::vector<std::vector<size_t>>&,                                      \
            std::vector<std::vector<size_t>>&                                       \
        );

    FOR_EACH_OP(INSTANTIATE_OP)

    #undef INSTANTIATE_T
    #undef INSTANTIATE_OP

} //namespace tensor