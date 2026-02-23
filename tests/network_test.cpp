#include "activation.hpp"
#include "matrix.hpp"
#include "network.hpp"
#include <cstdlib>
#include <gtest/gtest.h>
#include <vector>

TEST(NetworkTest, ForwardPassShape) {
    Matrix input(3, 1);
    input(0, 0) = 1.0;
    input(1, 0) = 0.5;
    input(2, 0) = -1.5;

    Network nn(input.getRows(), 5, 2, ActivationType::RELU);

    Matrix output = nn.forward(input);

    EXPECT_EQ(output.getRows(), 2);
    EXPECT_EQ(output.getCols(), 1);

    std::cout << "Network Output:" << std::endl;
    output.print();
}

TEST(NetworkTest, TrainXOR) {
    // 1. Setup Architecture: 2 inputs, 4 hidden nodes, 1 output
    Network nn(2, 4, 1, ActivationType::SIGMOID);

    // 2. Prepare XOR training data using your vector constructor
    std::vector<Matrix> inputs = {
        Matrix(std::vector<std::vector<double>>{{0.0}, {0.0}}),
        Matrix(std::vector<std::vector<double>>{{0.0}, {1.0}}),
        Matrix(std::vector<std::vector<double>>{{1.0}, {0.0}}),
        Matrix(std::vector<std::vector<double>>{{1.0}, {1.0}})};

    std::vector<Matrix> targets = {
        Matrix(std::vector<std::vector<double>>{{0.0}}),
        Matrix(std::vector<std::vector<double>>{{1.0}}),
        Matrix(std::vector<std::vector<double>>{{1.0}}),
        Matrix(std::vector<std::vector<double>>{{0.0}})};

    std::cout << "Training network..." << std::endl;

    // 3. Train for 50,000 epochs (iterations)
    int epochs = 50000;
    for (int i = 0; i < epochs; ++i) {
        // Pick a random training sample (0, 1, 2, or 3)
        int r = std::rand() % 4;
        nn.train(inputs[r], targets[r]);
    }

    std::cout << "\n--- Final XOR Predictions ---" << std::endl;

    // 4. Test the network after training
    for (int i = 0; i < 4; ++i) {
        Matrix result = nn.forward(inputs[i]);

        std::cout << "Input: [" << inputs[i](0, 0) << ", " << inputs[i](1, 0) << "] "
                  << "Target: " << targets[i](0, 0)
                  << " | Prediction: " << result(0, 0) << std::endl;
    }
}