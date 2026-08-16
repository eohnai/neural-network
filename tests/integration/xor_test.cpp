#include "matrix.hpp"
#include "matrix_test_helper.hpp"
#include "neural_network.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(XORTest, TrainXOR) {
    NeuralNetwork nn(2, 4, 1, ActivationType::SIGMOID, 42);

    std::vector<Matrix> trainingInputs = {
        test::makeMatrix<Matrix>(2, 1, {0.0, 0.0}), test::makeMatrix<Matrix>(2, 1, {0.0, 1.0}),
        test::makeMatrix<Matrix>(2, 1, {1.0, 0.0}), test::makeMatrix<Matrix>(2, 1, {1.0, 1.0})};

    std::vector<Matrix> trainingTargets = {
        test::makeMatrix<Matrix>(1, 1, {0.0}), test::makeMatrix<Matrix>(1, 1, {1.0}),
        test::makeMatrix<Matrix>(1, 1, {1.0}), test::makeMatrix<Matrix>(1, 1, {0.0})};

    constexpr int epochCount = 50000;
    for (int epoch = 0; epoch < epochCount; ++epoch) {
        for (int sampleIndex = 0; sampleIndex < 4; ++sampleIndex) {
            nn.train(trainingInputs[sampleIndex], trainingTargets[sampleIndex]);
        }
    }

    Matrix prediction00 = nn.forward(trainingInputs[0]);
    Matrix prediction01 = nn.forward(trainingInputs[1]);
    Matrix prediction10 = nn.forward(trainingInputs[2]);
    Matrix prediction11 = nn.forward(trainingInputs[3]);

    EXPECT_LT(prediction00(0, 0), 0.25);
    EXPECT_GT(prediction01(0, 0), 0.75);
    EXPECT_GT(prediction10(0, 0), 0.75);
    EXPECT_LT(prediction11(0, 0), 0.25);
}
