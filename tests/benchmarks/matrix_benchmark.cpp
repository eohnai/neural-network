// Standalone performance target; not part of the Google Test suite.
#include "activation.hpp"
#include "matrix.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

volatile double benchmarkChecksum = 0.0;

Matrix randomMatrix(int rows, int cols) {
    Matrix matrix(rows, cols);
    matrix.randomise();
    return matrix;
}

template <typename Operation>
void runBenchmark(const std::string &name, int warmupIterations, int measuredIterations, Operation operation) {
    for (int iteration = 0; iteration < warmupIterations; ++iteration) {
        Matrix result = operation();
        benchmarkChecksum += result(0, 0);
    }

    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < measuredIterations; ++iteration) {
        Matrix result = operation();
        benchmarkChecksum += result(0, 0);
    }
    const auto end = std::chrono::steady_clock::now();

    const double elapsedMilliseconds = std::chrono::duration<double, std::milli>(end - start).count();
    const double averageMilliseconds = elapsedMilliseconds / measuredIterations;

    std::cout << std::left << std::setw(56) << name
              << std::right << std::fixed << std::setprecision(3)
              << averageMilliseconds << " ms/op"
              << "  (" << measuredIterations << " iterations)\n";
}

}

int main() {
    std::cout << "NestedMatrix benchmark suite\n";
    std::cout << "All timings include result-matrix allocation.\n\n";

    const Matrix mnistInputWeights = randomMatrix(128, 784);
    const Matrix singleImage = randomMatrix(784, 1);
    const Matrix miniBatch = randomMatrix(784, 32);

    runBenchmark("MNIST inference: (128x784) * (784x1)", 20, 5000, [&] {
        return mnistInputWeights.dot(singleImage);
    });

    runBenchmark("MNIST minibatch: (128x784) * (784x32)", 5, 100, [&] {
        return mnistInputWeights.dot(miniBatch);
    });

    const Matrix squareLeft = randomMatrix(1000, 1000);
    const Matrix squareRight = randomMatrix(1000, 1000);
    runBenchmark("Large dense multiply: (1000x1000) * (1000x1000)", 1, 1, [&] {
        return squareLeft.dot(squareRight);
    });

    const Matrix elementwiseLeft = randomMatrix(4096, 4096);
    const Matrix elementwiseRight = randomMatrix(4096, 4096);
    runBenchmark("Element-wise addition: (4096x4096)", 1, 3, [&] {
        return elementwiseLeft.add(elementwiseRight);
    });

    runBenchmark("Element-wise Hadamard: (4096x4096)", 1, 3, [&] {
        return elementwiseLeft.hadamardProduct(elementwiseRight);
    });

    runBenchmark("Transpose: (4096x4096)", 1, 3, [&] {
        return elementwiseLeft.transpose();
    });

    runBenchmark("ReLU activation map: (4096x4096)", 1, 3, [&] {
        return elementwiseLeft.map(Activation::ReLU);
    });

    std::cout << "\nChecksum: " << benchmarkChecksum << "\n";
    return 0;
}
