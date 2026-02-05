#include "autograd/AutogradEngine.hpp"

namespace tensor
{
    void AutogradEngine::execute_graph(std::shared_ptr<Node> root_node, Tensor initial_grad)
    {
        // None
    }

    void AutogradEngine::backward(const std::vector<Tensor>& roots,
                          const std::vector<Tensor>& grad_outputs,
                          bool retain_graph,
                          bool create_graph)
    {
        // None
    }

} // namespace tensor