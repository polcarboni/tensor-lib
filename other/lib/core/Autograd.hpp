/*
AUTOGRAD NOTES
Autograd is based on a Dynamic Computational Graph.

The tensor class needs to store its history values.
Each operations needs to define also a backward pass
(addition, multiplication, matrix multiplication).

(e.g.) y = a * b. dy/db = a, dy/da = b;


Guide to building micrograd in Python:
https://www.youtube.com/watch?v=VMj-3S1tku0
(This implements sclar micrograd, then you can move to tensor grad)

Remember to implement no-grad mode (global flag or a scope guard).

--------------------------------------------------------------------

This file should define the 'Node' base class, with specific subclasses
for each operation.
(while the tensor.hpp class also need:
    - to store node for its history,  grad_fn = std::shared_ptr<Node>
    - grad: accumulated derivative
    - requires_grad: flag for activation
*/