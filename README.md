# tensor-lib
`tensor-lib` is a custom implementation of a tensor operation library based ATen/Pytorch.

### Relevant features:
+ **Type promotion**: Support for automatic type promotion in multi-operand operations. Supported types: `Bool`, `Int32`, `Int64`, `Float32`, `Float64`.
+ **Multi-Backend Support**: Support of different backend architectures: `CPU`, `CUDA`. (extensible dispatch system).
+ **Broadcasting**: Support of broadcasting logic for operations between different shaped tensors.
+ **Tensor Operations**: Implementation of the most relevant and used tensor operations.
+ **Visualization utilities**: Printing utility functions for visualization of tensor values and metadata.



**Purpose:** *This project serve as a learning tool to improve my understanding and practice in: Tensor operation libraries, C++ and CUDA programming languages, Software architecture and cross-compiler builds, Coding best practices.*

**Note**: This project is not intendend for production use.
Single GPU support only (sharding is not supported).

## Requirements

+ **CMake:** 3.22 or later
+ **C++ Compiler:** Supporting C++17 standard
+ **CUDA Toolkit:** 12.0.140 (optional, required for CUDA support)
+ **Catch2 (v3.12.0):** Unit testing framework (automatically fetched via FetchContent)


## Build Configuration

### Compiler Settings
+ **C++ Standard:** `C++17`
+ **Debug flags:** `-g3 -O0 -fno-omit-frame-pointer`
+ **Release flags:** `-O3`


### Optional Features

All of these featrues can be enabled/disabled via CMake options (all ON by default):
+ `ENABLE_TESTING`: Build unit tests with Catch2
+ `ENABLE_CLANG_TIDY`: Enable static analysis
+ `ENABLE_CUDA`: Enable NVIDIA CUDA support (auto-disabled if CUDA copmiler not found)
+ `ENABLE_SIMD`: Enable AVX2 SIMD Optimization (auto-disabled if not supported, still unused for implementations)


## Architectural overview

The main components/classes of the library are: 

#### Core classes:
+ **Types:** define allowed numerical types and utility functions. 
+ **Storage:** contains the actual data (used as shared_ptr by the tensorImpl class)
+ **Allocator:** allocates/deallocates memory based on the selected backend device type.
+ **TensorImpl:** contains pointer to data and metadata (shape, strides, dtype, ...) for the Tensor objects.
+ **Dispatchers:** called by the operation methods, initializes the iterator and calls the kernels
+ **TensorIterator:** definition of the operation logic.
+ **Tensor:** public facing API class.

#### Operation classes:
+ FillStorage operations
+ Unary operations
+ Binary operations
+ Reduction operations
+ *MatMul operations (to be implemented).*

Operations are separately defined for the different backend (CPU, CUDA).

For details refer to: [docs/architecture.md](docs/architecture.md)

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

<!-- For details refer to: [docs/api.md](docs/api.md) -->

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

For further examples see the [app/](app/) directory.


## Testing

Test suite is *under construction*, only base classes currently have tests. 

To run all currently defined tests:
```bash
make core_unit_tests
```

For running tests manually:
```bash
cd build/release
ctest --output-on-failure
```

### Planned features:
+ **Operations:** Matrix Multiplication, Reduction (CUDA backend).
+ **Classical ML:** Implementation of standard ML algorithms.
+ **Autograd:** Implementation of the automatic differentiation engine.
+ **Deep Learning:** Implementation of classes and DL Architectures.  
+ **Python bindings**


