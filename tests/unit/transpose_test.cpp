#include "matrix.hpp"
#include <gtest/gtest.h>

// TEST(TestSuiteName, IndividualTestName)
TEST(Transpose, Dimensions) {
    Matrix a(2, 3);
    Matrix c = a.transpose();

    EXPECT_EQ(c.getRows(), 3);
    EXPECT_EQ(c.getCols(), 2);
}

TEST(Transpose, ColumnToRow) {
    Matrix a({{1.0}, {2.0}, {3.0}});

    Matrix c = a.transpose();

    EXPECT_EQ(c(0, 0), 1.0);
    EXPECT_EQ(c(0, 1), 2.0);
    EXPECT_EQ(c(0, 2), 3.0);
}

TEST(Transpose, RowToColumn) {
    Matrix a({{1.0, 2.0, 3.0}});

    Matrix c = a.transpose();

    EXPECT_EQ(c(0, 0), 1.0);
    EXPECT_EQ(c(1, 0), 2.0);
    EXPECT_EQ(c(2, 0), 3.0);
}

TEST(Transpose, MatrixToMatrix) {
    Matrix a({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});

    Matrix c = a.transpose();

    EXPECT_EQ(c(0, 0), 1.0);
    EXPECT_EQ(c(0, 1), 3.0);
    EXPECT_EQ(c(0, 2), 5.0);
    EXPECT_EQ(c(1, 0), 2.0);
    EXPECT_EQ(c(1, 1), 4.0);
    EXPECT_EQ(c(1, 2), 6.0);
}