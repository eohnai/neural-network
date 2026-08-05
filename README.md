# C++ Neural Network (`cpp-nn`)

A simple, from-scratch implementation of a 3-layer feedforward neural network in C++. This project is built to understand the fundamentals of neural networks, backpropagation, and linear algebra without relying on external machine learning libraries.

## Features

- **Custom Matrix Library**: A lightweight matrix class supporting essential linear algebra operations (addition, subtraction, dot product, Hadamard product, transpose, and mapping functions).
- **3-Layer Architecture**: Configurable input, hidden, and output layers.
- **Activation Functions**: Support for ReLU and Sigmoid activation functions (with their derivatives).
- **Training & Inference**: Implements forward propagation for inference and backpropagation for training using Mean Squared Error (MSE) loss.
- **Unit Testing**: Comprehensive test suite using Google Test to ensure the correctness of matrix operations and network behavior.

## Prerequisites

- A C++17 compatible compiler (GCC, Clang, MSVC, etc.)
- CMake (>= 3.14)

## Building the Project

This project uses CMake for its build system. To build the project, run the following commands from the root directory:

```bash
mkdir build
cd build
cmake ..
make
```

This will generate the following executables in the `build` directory:

- `network`: The main application executable.
- `run_tests`: The Google Test executable for running unit tests.
- `run_benchmark`: Matrix-operation benchmark suite.
- `run_training_benchmark`: End-to-end neural-network training-throughput benchmark.

## Running the Application

After building, you can run the main application:

```bash
./network
```

## Running Tests

To ensure everything is working correctly, you can run the test suite:

```bash
./run_tests
```

## Running Benchmarks

To gather hardware optimization metrics, compile the project under **Release Mode**, then execute both benchmark targets:

```bash
cmake -S . -B build-nested -DCMAKE_BUILD_TYPE=Release
cmake --build build-nested
./build-nested/run_benchmark
./build-nested/run_training_benchmark
```

## Performance Benchmarks

To ensure the custom linear algebra engine optimizes cache locality and memory layout efficiently, a microbenchmark suite is used to track execution speeds under a Release compilation (`-O3`).

### Hardware Environment

- **CPU:** Apple M-Series (MacBook M2 Air)
- **Compiler:** Clang / GCC (C++17)

### Analysis & Observations

- **Nested Vector Overhead:** The baseline uses a nested vector structure (`std::vector<std::vector<double>>`). This introduces significant heap fragmentation because each row is allocated at a distinct memory address, resulting in persistent CPU L1/L2 cache misses during sequential loops.
- **Workloads:** The matrix suite includes large dense multiplication, element-wise operations, transpose, activation mapping, and MNIST-shaped single-image/minibatch multiplication. The training suite measures an end-to-end `784 -> 128 -> 10` training step.
- **Reporting:** Run the suite at least five times on a quiet machine and record the median for each workload before comparing implementations.

## Project Structure

```text
cpp-nn/
├── CMakeLists.txt      # CMake configuration file
├── include/            # Header files (.hpp)
│   ├── activation.hpp  # Activation functions
│   ├── matrix.hpp      # Public Matrix alias
│   ├── flat_matrix.hpp # Flat-vector Matrix implementation
│   ├── nested_matrix.hpp # Nested-vector Matrix implementation
│   └── neural_network.hpp # NeuralNetwork class
├── src/                # Source files (.cpp)
│   ├── activation.cpp
│   ├── main.cpp        # Main entry point
│   ├── nested_matrix.cpp
│   └── neural_network.cpp
├── reports/            # Benchmark methodology and saved measurements
│   ├── nested-matrix/
│   └── flat-matrix/
└── tests/              # Correctness and benchmark source files
    ├── benchmarks/     # Built as separate benchmark targets
    │   ├── matrix_benchmark.cpp
    │   └── training_benchmark.cpp
    ├── activation_test.cpp
    ├── add_test.cpp
    ├── dot_test.cpp
    ├── transpose_test.cpp
    └── xor_test.cpp

```

## License

This project is open-source and available under the MIT License.
