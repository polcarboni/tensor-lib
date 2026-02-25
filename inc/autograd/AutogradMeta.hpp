#pragma once
#include "core/TensorImpl.hpp"
#include "Node.hpp"
#include <memory>
#include <cstdint>

namespace tensor::grad
{
    // ------------------------------ AUTOGRAD META STRUCT  --------------------------------
    
    struct AutogradMeta {
        std::shared_ptr<TensorImpl> grad_ = nullptr;    /* Accumulated gradient */ 
        std::shared_ptr<Node> grad_fn_ = nullptr;       /* Operation that created the node */
        std::weak_ptr<Node> grad_accumulator_;
        uint32_t version_ = 0;

        std::unique_ptr<AutogradMeta> clone() const;
        
        bool is_leaf() const { return grad_fn_ == nullptr; }
    };

} // namespace tensor::grad