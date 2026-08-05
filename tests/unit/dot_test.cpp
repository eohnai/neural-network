#include "matrix.hpp"
#include <gtest/gtest.h>

// TEST(TestSuiteName, IndividualTestName)
TEST(DotProduct, ValidDimensions) {
    Matrix a(2, 3);
    Matrix b(3, 2);

    Matrix c = a.dot(b);

    EXPECT_EQ(c.getRows(), 2);
    EXPECT_EQ(c.getCols(), 2);
}

TEST(DotProduct, InvalidDimensions) {
    Matrix a(2, 2);
    Matrix b(1, 3);

    EXPECT_THROW(a.dot(b), std::invalid_argument);
}

TEST(DotProduct, InvalidDimensionsWithInitialisedValues) {
    Matrix a({{1.0, 2.0}, {3.0, 4.0}});
    Matrix b({{6.0, 5.0, 4.0}});

    EXPECT_THROW(a.dot(b), std::invalid_argument);
}

TEST(DotProduct, ScalarOutput) {
    Matrix a({{1.0, 2.0, 3.0}});
    Matrix b({{4.0}, {5.0}, {6.0}});

    Matrix c = a.dot(b);

    EXPECT_EQ(c(0, 0), 32.0);
}

TEST(DotProduct, MatrixOutput) {
    Matrix a({{1.0, 2.0}, {3.0, 4.0}});
    Matrix b({{4.0, 3.0}, {2.0, 1.0}});

    Matrix c = a.dot(b);

    EXPECT_EQ(c(0, 0), 8.0);
    EXPECT_EQ(c(0, 1), 5.0);
    EXPECT_EQ(c(1, 0), 20.0);
    EXPECT_EQ(c(1, 1), 13.0);
}

TEST(DotProduct, IdentityMatrixTest) {
    Matrix a({{1.0, 2.0, 3.0},
              {4.0, 5.0, 6.0}});

    Matrix identity({{1.0, 0.0, 0.0},
                     {0.0, 1.0, 0.0},
                     {0.0, 0.0, 1.0}});

    Matrix c = a.dot(identity);

    EXPECT_TRUE(c == a);
}