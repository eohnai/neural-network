#include "matrix.hpp"
#include "neural_network.hpp"
#include <gtest/gtest.h>

TEST(ForwardPassTest, OutputHasExpectedShape) {
    Matrix input(3, 1);
    input(0, 0) = 1.0;
    input(1, 0) = 0.5;
    input(2, 0) = -1.5;

    NeuralNetwork nn(input.getRows(), 5, 2, ActivationType::RELU);

    Matrix output = nn.forward(input);

    EXPECT_EQ(output.getRows(), 2);
    EXPECT_EQ(output.getCols(), 1);

}
