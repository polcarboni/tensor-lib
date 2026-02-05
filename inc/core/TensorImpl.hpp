#pragma once 
#include "Storage.hpp"
#include "autograd/AutogradMeta.hpp"
#include <vector>

namespace tensor
{
    // -------------------------------------------------------------------------------------------------------------  
    //                                        TENSOR IMPLEMENTATION CLASS
    // -------------------------------------------------------------------------------------------------------------

    // TODO: constructor to pass values to the storage (non-null initialization of vector)
    // TODO: check if view and clone are using same or new storage correctly

    struct TensorImpl
    {
        std::shared_ptr<Storage> storage_ = nullptr;
        ScalarType dtype_;
        Device device_;
        size_t offset_ = 0;                 /* Elements offset */
        std::vector<size_t> shape_;
        std::vector<size_t> strides_;
        size_t total_size_ = 0;             /* Number of elements */
        bool requires_grad_ = false;
        std::unique_ptr<AutogradMeta> autograd_meta_ = nullptr;

        // ---------------------------------------- HELPER FUNCTIONS ----------------------------------------
    
        void refresh_metadata();

        /* Helper: finds the actual memory address for logical indexing (useful in element-wise operations)*/
        size_t get_physical_offset(const std::vector<size_t>& indices) const;
    
        // ---------------------------------------- CONSTRUCTOR ----------------------------------------

        /* Impl from pointer to values */
        TensorImpl(const std::vector<size_t>& shape, ScalarType dtype, Device device, const void* src = nullptr);
        
        //TODO: requires methods for deep copy and shallow copy (same storage)


        // ---------------------------------------- GEOMETRIC FUNCTIONS ----------------------------------------

        bool is_contiguous() const;         /* Checks if the strides represent a contiguous representation of data */
        std::unique_ptr<TensorImpl> clone();
        std::unique_ptr<TensorImpl> contiguous() const;
        std::unique_ptr<TensorImpl> view(std::vector<size_t>& new_shape) const;
    };

} // namespace tensor