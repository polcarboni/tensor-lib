#pragma once
#include "Node.hpp"
#include <memory>

namespace tensor {

    struct Edge {
        std::shared_ptr<Node> function;
        uint32_t input_nr;

        Edge();
        Edge(std::shared_ptr<Node> function, uint32_t);

        bool is_valid() const;
    };
    
} // namespace tensor