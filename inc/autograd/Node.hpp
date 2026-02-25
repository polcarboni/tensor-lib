#pragma once
// #include "core/Tensor.hpp"
#include <vector>
#include <memory>
#include <atomic>
#include <cstdint>

namespace tensor
{
    class TensorImpl;
} // namespace tensor

/*
    The concrete implementations of this class contain the operator tensors and the
    overriden definition of the apply function.
    */
namespace tensor::grad
{
    class Edge;

    /**
     * Changing this for using a sequence_nr_ to a dependency count. Allows a correct use of multiple instances of the same node
     * with a better locality. 
     */
    class Node : public std::enable_shared_from_this<Node>{
    protected:
        std::vector<std::shared_ptr<Edge>> next_edges_;     /* gradient function inputs (parents) */
        // uint64_t sequence_nr_;                              /* backward pass execution order */
        // static std::atomic<uint64_t> next_sequence_nr_;     /* global counter for sequence_nr_ assignment */
        uint32_t dependency_count_;                         /* number of edges it depends on in the graph (apply when reaches zero)*/
        
        // TODO-fix: this can be a single Tensor to which contibutions are summed immediately.
        std::vector<TensorImpl> grad_buffer_;               /* grads required by the node*/

    public:
        Node();
        virtual ~Node();
        virtual std::vector<TensorImpl> apply(std::vector<TensorImpl>&& grads);

        void set_next_edges(std::vector<Edge>&& edges);
        void add_next_edge(Edge edge);
        const std::vector<Edge>& next_edges() const;
        uint64_t sequence_nr() const;

        size_t num_inputs() const;
    };


    /**
     * This node is created when the forward operation Add is used with Tensors with requires_grad_ = true.
     * Apply will call the backward version of the function when dependency_count_ = 0;
     */
    class AddBackward : public Node {
        std::vector<TensorImpl> apply(std::vector<TensorImpl>&& grads) override;
    };

} // namespace tensor::grad