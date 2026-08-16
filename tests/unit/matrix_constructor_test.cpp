#include "nested_matrix.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(NestedMatrixConstructor, RejectsNonPositiveDimensions) {
    EXPECT_THROW(NestedMatrix(0, 1), std::invalid_argument);
    EXPECT_THROW(NestedMatrix(1, 0), std::invalid_argument);
    EXPECT_THROW(NestedMatrix(-1, 1), std::invalid_argument);
}

TEST(NestedMatrixConstructor, RejectsEmptyNestedData) {
    const std::vector<std::vector<double>> emptyData;

    EXPECT_THROW(NestedMatrix{emptyData}, std::invalid_argument);
}

TEST(NestedMatrixConstructor, RejectsRowsWithoutColumns) {
    const std::vector<std::vector<double>> dataWithEmptyRow = {{}};

    EXPECT_THROW(NestedMatrix{dataWithEmptyRow}, std::invalid_argument);
}

TEST(NestedMatrixConstructor, RejectsRaggedNestedData) {
    const std::vector<std::vector<double>> raggedData = {{1.0, 2.0}, {3.0}};

    EXPECT_THROW(NestedMatrix{raggedData}, std::invalid_argument);
}

TEST(NestedMatrixConstructor, StoresValidNestedData) {
    NestedMatrix matrix({{1.0, 2.0}, {3.0, 4.0}});

    EXPECT_EQ(matrix.getRows(), 2);
    EXPECT_EQ(matrix.getCols(), 2);
    EXPECT_EQ(matrix(1, 0), 3.0);
}
