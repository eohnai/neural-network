#include "matrix.hpp"
#include <gtest/gtest.h>

TEST(HadamardProduct, MatrixOutput) {
    Matrix left({{1.0, 2.0}, {3.0, 4.0}});
    Matrix right({{5.0, 6.0}, {7.0, 8.0}});

    Matrix result = left.hadamardProduct(right);

    EXPECT_EQ(result(0, 0), 5.0);
    EXPECT_EQ(result(0, 1), 12.0);
    EXPECT_EQ(result(1, 0), 21.0);
    EXPECT_EQ(result(1, 1), 32.0);
}

TEST(HadamardProduct, RejectsMismatchedDimensions) {
    Matrix left(2, 2);
    Matrix right(2, 3);

    EXPECT_THROW(left.hadamardProduct(right), std::invalid_argument);
}
