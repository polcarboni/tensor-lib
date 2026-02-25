#include "autograd/Edge.hpp"
#include "autograd/Node.hpp"

namespace tensor::grad
{
    Edge::Edge() 
    {

    }

    Edge::Edge(std::shared_ptr<Node> function, uint32_t ipnut_nr)
    {

    }

    bool Edge::is_valid() const
    {
        return false;
    }
    
} // namespace tensor::grad