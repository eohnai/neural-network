// Standalone performance target; not part of the Google Test suite.
#include "matrix.hpp"
#include "neural_network.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>

int main() {
    constexpr int inputNodes = 784;
    constexpr int hiddenNodes = 128;
    constexpr int outputNodes = 10;
    constexpr int warmupIterations = 100;
    constexpr int measuredIterations = 1000;

    Matrix input(inputNodes, 1);
    input.randomise();

    Matrix target(outputNodes, 1);
    target(3, 0) = 1.0;

    NeuralNetwork network(inputNodes, hiddenNodes, outputNodes, ActivationType::SIGMOID);

    for (int iteration = 0; iteration < warmupIterations; ++iteration) {
        network.train(input, target);
    }

    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < measuredIterations; ++iteration) {
        network.train(input, target);
    }
    const auto end = std::chrono::steady_clock::now();

    const double elapsedMilliseconds = std::chrono::duration<double, std::milli>(end - start).count();
    const double millisecondsPerStep = elapsedMilliseconds / measuredIterations;
    const double stepsPerSecond = 1000.0 / millisecondsPerStep;

    const Matrix prediction = network.forward(input);

    std::cout << "NestedMatrix training benchmark\n";
    std::cout << "Network: " << inputNodes << " -> " << hiddenNodes << " -> " << outputNodes << "\n";
    std::cout << "Workload: single-sample training step (forward pass + backpropagation)\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Average: " << millisecondsPerStep << " ms/step\n";
    std::cout << "Throughput: " << stepsPerSecond << " steps/s\n";
    std::cout << "Verification output: " << prediction(3, 0) << "\n";

    return 0;
}
