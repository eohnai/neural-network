#include "matrix.hpp"
#include <gtest/gtest.h>

TEST(Subtract, MatrixOutput) {
    Matrix left({{5.0, 4.0}, {3.0, 2.0}});
    Matrix right({{1.0, 2.0}, {3.0, 4.0}});

    Matrix result = left.subtract(right);

    EXPECT_EQ(result(0, 0), 4.0);
    EXPECT_EQ(result(0, 1), 2.0);
    EXPECT_EQ(result(1, 0), 0.0);
    EXPECT_EQ(result(1, 1), -2.0);
}

TEST(Subtract, RejectsMismatchedDimensions) {
    Matrix left(2, 2);
    Matrix right(1, 2);

    EXPECT_THROW(left.subtract(right), std::invalid_argument);
}
