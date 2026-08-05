#include "matrix.hpp"
#include <gtest/gtest.h>

// TEST(TestSuiteName, IndividualTestName)
TEST(Add, ValidDimensions) {
    Matrix a(2, 3);
    Matrix b(2, 3);

    Matrix c = a.add(b);

    EXPECT_EQ(c.getRows(), 2);
    EXPECT_EQ(c.getCols(), 3);
}

TEST(Add, InvalidDimensions) {
    Matrix a(3, 2);
    Matrix b(2, 3);

    EXPECT_THROW(a.add(b), std::invalid_argument);
}

TEST(Add, InvalidDimensionsWithInitialisedValues) {
    Matrix a({{1.0, 2.0}, {3.0, 4.0}});
    Matrix b({{6.0, 5.0, 4.0}});

    EXPECT_THROW(a.add(b), std::invalid_argument);
}

TEST(Add, VectorOutput) {
    Matrix a({{1.0}, {2.0}, {3.0}});
    Matrix b({{4.0}, {5.0}, {6.0}});

    Matrix c = a.add(b);

    EXPECT_EQ(c(0, 0), 5.0);
    EXPECT_EQ(c(1, 0), 7.0);
    EXPECT_EQ(c(2, 0), 9.0);
}

TEST(Add, MatrixOutput) {
    Matrix a({{1.0, 2.0}, {3.0, 4.0}});
    Matrix b({{4.0, 3.0}, {2.0, 1.0}});

    Matrix c = a.add(b);

    EXPECT_EQ(c(0, 0), 5.0);
    EXPECT_EQ(c(0, 1), 5.0);
    EXPECT_EQ(c(1, 0), 5.0);
    EXPECT_EQ(c(1, 1), 5.0);
}