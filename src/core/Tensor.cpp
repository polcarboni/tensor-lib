#include "core/Tensor.hpp"

namespace tensor
{
    // ---------------------------- CONSTRUCTORS ---------------------------- 

    Tensor::Tensor()
        : pimpl_(nullptr) {}
    
    Tensor::Tensor(const std::vector<size_t>& shape) {}
    
    Tensor::Tensor(std::unique_ptr<TensorImpl> impl)
        : pimpl_(std::move(impl)) {}

    Tensor::Tensor(const std::vector<size_t>& shape, ScalarType dtype, Device device)
        : pimpl_(nullptr) 
    {
        // Implementation would go here
    }

    Tensor::~Tensor() = default;

    Tensor::Tensor(const Tensor& other) : pimpl_(nullptr)
    {
        // Shallow copy logic would go here
    }

    Tensor Tensor::clone() const
    {
        return Tensor(); // placeholder
    }

    // template <typename T>
    // Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<T>& values, Device device);

    // ---------------------------- ACCESSORS ---------------------------- 
    
    TensorImpl* Tensor::impl() const
    {
        return pimpl_.get();
    }
    
    const std::vector<size_t>& Tensor::shape() const
    {
        static std::vector<size_t> dummy;
        return dummy;
    }
    
    const std::vector<size_t>& Tensor::strides() const
    {
        static std::vector<size_t> dummy;
        return dummy;
    }
    
    ScalarType Tensor::dtype() const
    {
        return ScalarType::Float32;
    }
    
    Device Tensor::device() const
    {
        return {DeviceType::CPU, 0};
    }
    
    size_t Tensor::size() const
    {
        return 0;
    }
    
    size_t Tensor::dims() const
    {
        return 0;
    }
    
    // ---------------------------- GRAD ACCESSORS ---------------------------- 

    bool Tensor::requires_grad() const
    {
        return false;
    }

    void Tensor::set_requires_grad(bool r)
    {
        // empty
    }

    void Tensor::backward(const Tensor& gradient, bool retain_graph, bool create_graph)
    {
        // empty
    }

    Tensor Tensor::grad() const {
        Tensor dummy = Tensor();
        return dummy;
    }

    void Tensor::set_grad(Tensor grad)
    {
        // NONE    
    }
    
    std::shared_ptr<Node> Tensor::grad_fn() const
    {
        auto a = std::shared_ptr<Node> {};
        return a;
    }
    
    void Tensor::set_grad_fn(std::shared_ptr<Node> fn)
    {
        // NONE    
    }    

    bool Tensor::is_leaf() const
    {
        return false;
    }

    uint32_t Tensor::output_nr() const
    {
        uint32_t a = 0;
        return a;
    }

    // ---------------------------- UTILITY FUNCTIONS ---------------------------- 

    Tensor Tensor::contiguous() const
    {
        return Tensor();
    }

    bool Tensor::is_contiguous() const
    {
        return true;
    }

    Tensor Tensor::view(std::vector<size_t>& shape) const
    {
        return Tensor();
    }

    Tensor Tensor::reshape(std::vector<size_t>& shape) const
    {
        return Tensor();
    }

} // namespace tensor
