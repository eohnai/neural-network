#include "../include/activation.hpp"
#include "../include/matrix.hpp"
#include <gtest/gtest.h>

// TEST(TestSuiteName, IndividualTestName)
TEST(ActivationTest, ReLU) {
    Matrix a({{-0.5, 2.0}, {0.5, -5.0}});

    a.activation(Activation::ReLU);

    EXPECT_EQ(a(0, 0), 0.0);
    EXPECT_EQ(a(0, 1), 2.0);
    EXPECT_EQ(a(1, 0), 0.5);
    EXPECT_EQ(a(1, 1), 0.0);
}

TEST(ActivationTest, sigmoid) {
    Matrix a({{-0.5, 2.0}, {0.5, -5.0}});

    a.activation(Activation::sigmoid);

    EXPECT_EQ(a(0, 0), 1.0 / (1.0 + std::exp(-(-0.5))));
    EXPECT_EQ(a(0, 1), 1.0 / (1.0 + std::exp(-2.0)));
    EXPECT_EQ(a(1, 0), 1.0 / (1.0 + std::exp(-0.5)));
    EXPECT_EQ(a(1, 1), 1.0 / (1.0 + std::exp(-(-5.0))));
}