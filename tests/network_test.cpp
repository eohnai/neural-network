#include "../include/matrix.hpp"
#include "../include/network.hpp"
#include <gtest/gtest.h>

TEST(NetworkTest, ForwardPassShape) {
    Matrix input(3, 1);
    input(0, 0) = 1.0;
    input(1, 0) = 0.5;
    input(2, 0) = -1.5;

    Network nn(input.getRows(), 5, 2);

    Matrix output = nn.forward(input);

    EXPECT_EQ(output.getRows(), 2);
    EXPECT_EQ(output.getCols(), 1);

    std::cout << "Network Output:" << std::endl;
    output.print();
}