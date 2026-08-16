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

After building, `network` trains a `784 -> 128 -> 10` classifier using the local MNIST IDX files in `data/mnist/raw/`, then evaluates it against the test split. By default it trains for one epoch on all training samples:

```bash
./network
```

You can provide the epoch count, a cap on training samples, and an alternate data directory. This quick run is useful for checking that the pipeline works without waiting for a full epoch:

```bash
./network 1 1000
```

```text
./network [epochs] [max-training-samples] [data-directory]
```

## Running Tests

To ensure everything is working correctly, you can run the test suite:

```bash
./run_tests
```

## Running Benchmarks

To gather hardware optimization metrics, compile the project under **Release Mode**, then execute both benchmark targets:

```bash
cmake -S . -B build-nested -DCMAKE_BUILD_TYPE=Release -DMATRIX_IMPLEMENTATION=NESTED
cmake --build build-nested --target run_benchmark run_training_benchmark
./build-nested/run_benchmark
./build-nested/run_training_benchmark
```

Build the flat implementation in a separate directory so the two executables can be compared directly:

```bash
cmake -S . -B build-flat -DCMAKE_BUILD_TYPE=Release -DMATRIX_IMPLEMENTATION=FLAT
cmake --build build-flat --target run_benchmark run_training_benchmark
./build-flat/run_benchmark
./build-flat/run_training_benchmark
```

## Performance Benchmarks

To ensure the custom linear algebra engine optimizes cache locality and memory layout efficiently, a microbenchmark suite is used to track execution speeds under a Release compilation (`-O3`).

### Hardware Environment

- **CPU:** Apple M-Series (MacBook M2 Air)
- **Compiler:** Clang / GCC (C++17)

### MNIST training results

The end-to-end measurements below use a `784 -> 128 -> 10` network, fixed initialization and shuffle seeds, single-sample gradient descent, and the standard 60,000-image MNIST training split. Test accuracy is evaluated against the 10,000-image test split after training.

#### NestedMatrix baseline

| Command | Training workload | Training time | Test accuracy |
| --- | --- | ---: | ---: |
| `./build-nested/network 1 1000` | 1 epoch, first 1,000 samples | 0.473 s | 19.22% |
| `./build-nested/network 1` | 1 epoch, 60,000 samples; median of 5 independent runs | 28.456 s | 68.52% |
| `./build-nested/network 5` | 5 epochs, 60,000 samples each | 139.400 s total; 27.880 s/epoch | 82.94% |
| `./build-nested/network 10` | 10 epochs, 60,000 samples each | 286.427 s total; 28.643 s/epoch | 84.18% |
| `./build-nested/network 20` | 20 epochs, 60,000 samples each | 574.838 s total; 28.742 s/epoch | 85.54% |

#### FlatMatrix results

| Command | Training workload | Training time | Test accuracy |
| --- | --- | ---: | ---: |
| `./build-flat/network 1` | 1 epoch, 60,000 samples; median of 5 independent runs | 18.789 s | 68.52% |
| `./build-flat/network 5` | 5 epochs, 60,000 samples each | 95.575 s total; 19.115 s/epoch | 82.94% |
| `./build-flat/network 10` | 10 epochs, 60,000 samples each | 190 s total; 19.0 s/epoch | 84.18% |
| `./build-flat/network 20` | 20 epochs, 60,000 samples each | 381.547 s total; 19.077 s/epoch | 85.54% |

#### 5-epoch comparison

| Implementation | Total training time | Average per epoch | Test accuracy |
| --- | ---: | ---: | ---: |
| NestedMatrix | 139.400 s | 27.880 s | 82.94% |
| FlatMatrix | 95.575 s | 19.115 s | 82.94% |

For this workload, `FlatMatrix` completed the five-epoch run about **1.46x faster**, reducing training time by about **31.4%** while reaching the same deterministic test accuracy.

#### 10-epoch comparison

| Implementation | Total training time | Average per epoch | Test accuracy |
| --- | ---: | ---: | ---: |
| NestedMatrix | 286.427 s | 28.643 s | 84.18% |
| FlatMatrix | 190 s | 19.0 s | 84.18% |

For this workload, `FlatMatrix` completed the 10-epoch run about **1.51x faster**, reducing training time by about **33.7%** while reaching the same deterministic test accuracy.

#### 20-epoch comparison

| Implementation | Total training time | Average per epoch | Test accuracy |
| --- | ---: | ---: | ---: |
| NestedMatrix | 574.838 s | 28.742 s | 85.54% |
| FlatMatrix | 381.547 s | 19.077 s | 85.54% |

For this workload, `FlatMatrix` completed the 20-epoch run about **1.51x faster**, reducing training time by about **33.6%** while reaching the same deterministic test accuracy.

#### Repeated one-epoch comparison

| Implementation | Median training time from 5 fresh runs | Test accuracy |
| --- | ---: | ---: |
| NestedMatrix | 28.456 s | 68.52% |
| FlatMatrix | 18.789 s | 68.52% |

Across five independent one-epoch runs, `FlatMatrix` was about **1.51x faster** and reduced median training time by about **34.0%**, with identical accuracy.

### Analysis & Observations

- **Nested Vector Overhead:** `NestedMatrix` uses `std::vector<std::vector<double>>`, which has one separately allocated payload per row. Tall column vectors such as `784 x 1` therefore contain many tiny allocations and require pointer indirection, while `FlatMatrix` stores all values in one contiguous allocation. This can increase allocation cost and reduce locality, but elapsed-time results alone do not prove a specific cache-miss rate.
- **Workloads:** The matrix suite includes large dense multiplication, element-wise operations, transpose, activation mapping, and MNIST-shaped single-image/minibatch multiplication. The training suite measures an end-to-end `784 -> 128 -> 10` training step.
- **Reporting:** Run the suite at least five times on a quiet machine and record the median for each workload before comparing implementations.

## Roadmap

- [x] Implement and benchmark `NestedMatrix` using `std::vector<std::vector<double>>`.
- [x] Implement and benchmark contiguous `FlatMatrix` storage using `std::vector<double>`.
- [x] Validate that both implementations produce identical deterministic MNIST accuracy.
- [ ] Decouple dataset conversion and network training from the build-time `Matrix` alias so both precompiled implementations can be selected at runtime.
- [ ] Add a terminal user interface for selecting a dataset, matrix implementation, epoch count, and sample limit.
- [ ] Profile the end-to-end training path to identify the true CPU bottlenecks.
- [ ] Add SIMD optimizations to the measured hot loops.
- [ ] Explore a Metal GPU backend after the CPU implementation is profiled and optimized.
- [ ] Improve model accuracy after the matrix-storage performance comparison is established, without mixing model-quality changes into the storage benchmark.

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
