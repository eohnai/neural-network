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
```

## License

This project is open-source and available under the MIT License.
