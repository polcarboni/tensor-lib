# tensor-lib
tensor-lib is a custom implementation of a tensor operation library based on ATen/Pytorch.

Relevant features:
+ Type promotions in multiple operand oerations. Supported types: Bool, Int32, Int64, Float32, Float64 (Double).
+ Support of different backends architectrues: CPU, CUDA. (With extensible dispatch system).
+ Support of broadcasting for operations between different shaped tensors.
+ Implementation of the most relevant and used tensor operations.


This library serves the purposes of deepen my understanding and practice of: Tensor operation libraries, C++ and CUDA programming language, Software architecture and cross compiler builds, good coding practices (without intentions of making it production ready). 

### Build configuration:
+ C++17
+ CUDAToolkit: 12.0.140

## API use examples

```cpp
#include <iostream>
#include "TensorLib.hpp"

using namespace tensor;

int main() {
    Tensor a({2, 3}, ScalarType::Float32, {DeviceType::CUDA, 0});
    a.fill_arange(1.0, 1.0);
    
    Tensor b({2, 3}, ScalarType::Float32, {DeviceType::CUDA, 0});
    b.fill_arange(6.0, -1.0);
    
    Tensor sum = a.add(b);
    std::cout << sum << "\n";
}
```

### Further things to be implemented:
+ Operations: Matrix Multiplication, Reduction (CUDA backend).
+ Classical ML algorithms.
+ Autograd engine.
+ Deep Learning architectures.  
+ Python bindings.