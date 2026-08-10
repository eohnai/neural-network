#include "matrix.hpp"
#include "neural_network.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(NetworkValidation, TrainRejectsTargetWithWrongRowCount) {
    NeuralNetwork network(2, 4, 1, ActivationType::SIGMOID, 42);
    Matrix input(std::vector<std::vector<double>>{{0.0}, {1.0}});
    Matrix target(2, 1);

    EXPECT_THROW(network.train(input, target), std::invalid_argument);
}

TEST(NetworkValidation, TrainRejectsTargetWithWrongColumnCount) {
    NeuralNetwork network(2, 4, 1, ActivationType::SIGMOID, 42);
    Matrix input(std::vector<std::vector<double>>{{0.0}, {1.0}});
    Matrix target(1, 2);

    EXPECT_THROW(network.train(input, target), std::invalid_argument);
}
