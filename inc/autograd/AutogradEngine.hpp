#pragma once 
#include "core/TensorImpl.hpp"
#include <vector>

namespace tensor::grad
{
    // class TensorImpl;
    class Node;

    struct AutogradEngine {
    private:
        /**
         * Performs the traversal of the DAG (directed acyclical graph):
         * graph discovery, topological scheduling, backward esecution and gradient propagation (in a single dynamic pass)
         */
        static void execute_graph(std::shared_ptr<Node> root_node, TensorImpl initial_grad)
        {
             /**
              * Graph discovery and execution (dynamic autograd)
              * 
              * Calling the backward on a node will trigger the use of the apply function. Computing the parents grad tensors.
              * 
              * The execution logic is delegated to the single nodes, as they keep track of the required amount of dependencies
              * (prevents a premature backprop for skip connections).
              * 
              * Recursive call:
              * loop the call of the same function over the elements of root_node.next_edges_ data member.
              */
        }
    
    public:
        static void backward(const std::vector<TensorImpl>& roots,
                             const std::vector<TensorImpl>& grad_outputs,
                             bool retain_graph = false,
                             bool create_graph = false)
        {
            /**
             * Public API function that calls the execut_graph function on the last node of a DAG.
             * 
             * I assume that the 2 bool flags are used for when a graph is already created or for when is not,
             * but if so it would not have much sense using 2 separate flags (what is the alternative?).
             */
        }
    };

} // namespace tensor::grad