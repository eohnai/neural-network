#include "flat_matrix.hpp"

#include <gtest/gtest.h>

#include <random>
#include <stdexcept>
#include <vector>

TEST(FlatMatrix, ConstructsZeroMatrixAndValidatesData) {
    FlatMatrix matrix(2, 3);

    EXPECT_EQ(matrix.getRows(), 2);
    EXPECT_EQ(matrix.getCols(), 3);
    EXPECT_DOUBLE_EQ(matrix(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(matrix(1, 2), 0.0);

    matrix(1, 2) = 9.0;
    const FlatMatrix &readOnlyMatrix = matrix;
    EXPECT_DOUBLE_EQ(readOnlyMatrix(1, 2), 9.0);

    EXPECT_THROW(FlatMatrix(0, 1), std::invalid_argument);
    EXPECT_THROW(FlatMatrix(1, -1), std::invalid_argument);

    const std::vector<double> incompleteValues{1.0, 2.0, 3.0};
    EXPECT_THROW(FlatMatrix(2, 2, incompleteValues), std::invalid_argument);
}

TEST(FlatMatrix, PerformsElementWiseOperationsOnRectangularMatrices) {
    const FlatMatrix left(2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    const FlatMatrix right(2, 3, {6.0, 5.0, 4.0, 3.0, 2.0, 1.0});

    const FlatMatrix sum = left.add(right);
    const FlatMatrix difference = left.subtract(right);
    const FlatMatrix product = left.hadamardProduct(right);

    EXPECT_EQ(sum, FlatMatrix(2, 3, {7.0, 7.0, 7.0, 7.0, 7.0, 7.0}));
    EXPECT_EQ(difference, FlatMatrix(2, 3, {-5.0, -3.0, -1.0, 1.0, 3.0, 5.0}));
    EXPECT_EQ(product, FlatMatrix(2, 3, {6.0, 10.0, 12.0, 12.0, 10.0, 6.0}));

    EXPECT_THROW(left.add(FlatMatrix(3, 2)), std::invalid_argument);
}

TEST(FlatMatrix, MultipliesAndTransposesRectangularMatrices) {
    const FlatMatrix left(2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    const FlatMatrix right(3, 2, {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});

    const FlatMatrix result = left.dot(right);
    EXPECT_EQ(result, FlatMatrix(2, 2, {58.0, 64.0, 139.0, 154.0}));

    const FlatMatrix transposed = left.transpose();
    EXPECT_EQ(transposed, FlatMatrix(3, 2, {1.0, 4.0, 2.0, 5.0, 3.0, 6.0}));
}

TEST(FlatMatrix, MapsValuesAndRandomisesDeterministicallyWithSeed) {
    const FlatMatrix matrix(2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    const FlatMatrix doubled = matrix.map([](double value) { return value * 2.0; });
    EXPECT_EQ(doubled, FlatMatrix(2, 3, {2.0, 4.0, 6.0, 8.0, 10.0, 12.0}));

    std::mt19937_64 firstGenerator(42);
    std::mt19937_64 secondGenerator(42);
    FlatMatrix first(2, 3);
    FlatMatrix second(2, 3);
    first.randomise(firstGenerator);
    second.randomise(secondGenerator);

    EXPECT_EQ(first, second);
}
