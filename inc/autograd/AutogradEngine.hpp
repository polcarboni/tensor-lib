#pragma once 
#include "core/Tensor.hpp"

namespace tensor
{

    struct AutogradEngine {
    private:
        static void execute_graph(std::shared_ptr<Node> root_node, Tensor initial_grad);
    
    public:
        static void backward(const std::vector<Tensor>& roots,
                             const std::vector<Tensor>& grad_outputs,
                             bool retain_graph = false,
                             bool create_graph = false);
    };

} // namespace tensor