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

This will generate two executables in the `build` directory:

- `network`: The main application executable.
- `run_tests`: The Google Test executable for running unit tests.

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

To gather hardware optimization metrics, compile the project under **Release Mode** to enable maximum compiler optimization flags (`-O3`), then execute the benchmark target:

````bash
cmake -DCMAKE_BUILD_TYPE=Release ..
make
./run_benchmark

## Performance Benchmarks

To ensure the custom linear algebra engine optimizes cache locality and memory layout efficiently, a microbenchmark suite is used to track execution speeds under a Release compilation (`-O3`).

### Hardware Environment
- **CPU:** Apple M-Series (MacBook M2 Air)
- **Compiler:** Clang / GCC (C++17)

### Implementation Comparison

| Operation | Matrix Dimensions | 2D Vector Layout (Baseline) | 1D Flat Vector Layout (Optimized) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **Matrix Multiplication (`dot`)** | $1000 \times 1000$ | 5685 ms | *TBD* | *TBD* |
| **Matrix Addition (`add`)** | $10000 \times 10000$ | 1401 ms | *TBD* | *TBD* |

### Analysis & Observations
- **2D Vector Overhead:** The baseline uses a nested vector structure (`std::vector<std::vector<double>>`). This introduces significant heap fragmentation because each row is allocated at a distinct memory address, resulting in persistent CPU L1/L2 cache misses during sequential loops.

## Project Structure

```text
cpp-nn/
├── CMakeLists.txt      # CMake configuration file
├── include/            # Header files (.hpp)
│   ├── activation.hpp  # Activation functions
│   ├── matrix.hpp      # Custom Matrix class
│   └── network.hpp     # Neural Network class
├── src/                # Source files (.cpp)
│   ├── activation.cpp
│   ├── main.cpp        # Main entry point
│   ├── matrix.cpp
│   └── network.cpp
└── tests/              # Google Test files
    ├── activation_test.cpp
    ├── add_test.cpp
    ├── dot_test.cpp
    ├── network_test.cpp
    └── transpose_test.cpp
````

## License

This project is open-source and available under the MIT License.
