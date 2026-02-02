#pragma once
#include "core/Tensor.hpp"

namespace tensor
{
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

} // namespace tensor