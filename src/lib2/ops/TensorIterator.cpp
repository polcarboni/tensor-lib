#include "ops/TensorIterator.hpp"

namespace tensor
{

    // --------------------------------- TENSOR ITERATOR CONFIG ---------------------------------

    TensorIteratorConfig& TensorIteratorConfig::add_input(const Tensor& t)
    {

    }

    TensorIteratorConfig& TensorIteratorConfig::add_output(Tensor& t)
    {

    }

    TensorIteratorConfig& TensorIteratorConfig::add_reduction(bool b)
    {

    }

    // --------------------------------- TENSOR ITERATOR ---------------------------------

    void TensorIterator::collapse_dims()
    {

    }

    TensorIterator TensorIterator::build(const TensorIteratorConfig& config)
    {

    }

    Device TensorIterator::device() const
    {

    }

    ScalarType TensorIterator::common_dtype() const
    {

    }

    size_t TensorIterator::num_elements() const
    {

    }

    bool TensorIterator::is_contiguous() const
    {

    }

    int TensorIterator::ninputs() const
    {

    }
    
    int TensorIterator::noutputs() const
    {

    }

    void* TensorIterator::data_ptr(int arg) const
    {

    }

    bool TensorIterator::is_trivial_1d() const
    {

    }

    size_t TensorIterator::reduction_block_size() const
    {
        
    }

} // namespace tensor