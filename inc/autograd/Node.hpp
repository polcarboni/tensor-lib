#pragma once
#include "core/Tensor.hpp"
#include <vector>
#include <memory>
#include <atomic>
#include <cstdint>

/*
    The concrete implementations of this class contain the operator tensors and the
    overriden definition of the apply function.
    */
namespace tensor
{
    class Edge;

    class Node : public std::enable_shared_from_this<Node>{
    protected:
        std::vector<std::shared_ptr<Edge>> next_edges_;
        uint64_t sequence_nr_;
        static std::atomic<uint64_t> next_sequence_nr_;

    public:
        Node();
        virtual ~Node();
        virtual std::vector<Tensor> apply(std::vector<Tensor>&& grads);

        void set_next_edges(std::vector<Edge>&& edges);
        void add_next_edge(Edge edge);
        const std::vector<Edge>& next_edges() const;
        uint64_t sequence_nr() const;

        size_t num_inputs() const;
    };

} // namespace tensor