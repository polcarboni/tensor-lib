# tensor-lib
tensor-lib is a custom implementation of a tensor operation library based on ATen/Pytorch.

### Relevant features:
+ **Type promotion**: Support for automatic type promotion in multi-operand operations. Supported types: Bool, Int32, Int64, Float32, Float64 (Double).
+ **Multi-Backend Support**: Support of different backends architectrues: CPU, CUDA. (With extensible dispatch system).
+ **Broadcasting**: Support of broadcasting for operations between different shaped tensors.
+ **Tensor Operations**: Implementation of the most relevant and used tensor operations.
+ **Visualization utilities**: Printing utility functions for visualization of tensor values and metadata.



This project serve as a learning project to improve my understanding and practice in: Tensor operation libraries, C++ and CUDA programming languages, Software architecture and cross compiler builds, Coding best practices

**Note**: This is not intendend for production use.
Single GPU support only (sharding is not supported).

## Requirements

+ **CMake:** 3.22 or later
+ **C++ Compiler:** Supporting C++17 standard
+ **CUDA Toolkit:** 12.0.140 (optional, required for CUDA support)
+ **Catch2 (v3.12.0):** Unit testing framework (automatically fetched via FetchContent)


## Build Configuration

### Compiler Settings
+ **C++ Standard:** C++17
+ **Debug flags:** -g3 -O0 -fno-omit-frame-pointer
+ **Release flags:** -O3


### Optional Features

All of these featrues can be enabled/disabled via CMake options (all ON by default):
+ `ENABLE_TESTING`: Build unit tests with Catch2
+ `ENABLE_CLANG_TIDY`: Enable static analysis
+ `ENABLE_CUDA`: Enable NVIDIA CUDA support (auto-disabled if CUDA copmiler not found)
+ `ENABLE_SIMD`: Enable AVX2 SIMD Optimization (auto-disabled if not supported, still unused for implementations)


## Architectural overview

+ Storage Model
+ Dispatch system
+ Kernel abstraction
+ Memory mangement ?

For details refer to: docs/architecture.md

## API Documentation

#### Core classes
+ Tensor
+ TensorImpl
+ TensorIterator

#### Key methods
+ Creation
+ Filling
+ Unary operations
+ ...

For details refer to: [docs/api.md](docs/api.md)

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

For further examples see the [app/ directory](app/).


## Testing

Test suite is under construction, only base classes currently have tests. 

```
HOW TO RUN TESTS
```


### Planned features:
+ Operations: Matrix Multiplication, Reduction (CUDA backend).
+ Classical ML algorithms.
+ Autograd engine.
+ Deep Learning architectures.  
+ Python bindings.


