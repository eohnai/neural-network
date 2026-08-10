#include "matrix.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(MatrixConstructor, RejectsNonPositiveDimensions) {
    EXPECT_THROW(Matrix(0, 1), std::invalid_argument);
    EXPECT_THROW(Matrix(1, 0), std::invalid_argument);
    EXPECT_THROW(Matrix(-1, 1), std::invalid_argument);
}

TEST(MatrixConstructor, RejectsEmptyNestedData) {
    const std::vector<std::vector<double>> emptyData;

    EXPECT_THROW(Matrix{emptyData}, std::invalid_argument);
}

TEST(MatrixConstructor, RejectsRowsWithoutColumns) {
    const std::vector<std::vector<double>> dataWithEmptyRow = {{}};

    EXPECT_THROW(Matrix{dataWithEmptyRow}, std::invalid_argument);
}

TEST(MatrixConstructor, RejectsRaggedNestedData) {
    const std::vector<std::vector<double>> raggedData = {{1.0, 2.0}, {3.0}};

    EXPECT_THROW(Matrix{raggedData}, std::invalid_argument);
}

TEST(MatrixConstructor, StoresValidNestedData) {
    Matrix matrix({{1.0, 2.0}, {3.0, 4.0}});

    EXPECT_EQ(matrix.getRows(), 2);
    EXPECT_EQ(matrix.getCols(), 2);
    EXPECT_EQ(matrix(1, 0), 3.0);
}
