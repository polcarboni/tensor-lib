#pragma once
// #include "core/Tensor.hpp"
// #include "Node.hpp"
#include <memory>
#include <cstdint>

namespace tensor
{
    // ------------------------------ AUTOGRAD META STRUCT  --------------------------------
    
    class Tensor;
    class Node;
    
    struct AutogradMeta {
        std::shared_ptr<Tensor> grad_ = nullptr;    
        std::shared_ptr<Node> grad_fn_ = nullptr;
        std::weak_ptr<Node> grad_accumulator_;
        uint32_t version_ = 0;
    };

} // namespace tensor