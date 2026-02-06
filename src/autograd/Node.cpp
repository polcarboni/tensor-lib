#include "autograd/Node.hpp"
#include "autograd/Edge.hpp"

namespace tensor
{

    Node::Node() 
    {
        // None
    }

    Node::~Node()
    {
        // None
    }

    std::vector<Tensor> Node::apply(std::vector<Tensor>&& grads)
    {
        auto a = std::vector<Tensor> {};
        return a;
    }

    void Node::set_next_edges(std::vector<Edge>&& edges)
    {
        // None
    }
    
    void Node::add_next_edge(Edge edge)
    {   
        // None
    }
    
    const std::vector<Edge>& Node::next_edges() const
    {
        auto a = std::vector<Edge> {};
        return a;
    }
    
    uint64_t Node::sequence_nr() const
    {
        uint64_t a = 0;
        return a;
    }

    size_t Node::num_inputs() const
    {
        size_t a = 0;
        return a;
    }
    
} // namespace tensor