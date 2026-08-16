#pragma once

#include "flat_matrix.hpp"
#include "nested_matrix.hpp"

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace test {

// Builds either matrix implementation from row-major values so that shared
// operation tests do not depend on a storage-specific constructor.
template <typename MatrixT>
MatrixT makeMatrix(int rowCount, int colCount, const std::vector<double> &values) {
    if (rowCount < 1 || colCount < 1) {
        throw std::invalid_argument("Test matrices must have positive dimensions.");
    }

    const std::size_t expectedValueCount =
        static_cast<std::size_t>(rowCount) * static_cast<std::size_t>(colCount);
    if (values.size() != expectedValueCount) {
        throw std::invalid_argument("Test matrix value count does not match its dimensions.");
    }

    if constexpr (std::is_same_v<MatrixT, FlatMatrix>) {
        return FlatMatrix(rowCount, colCount, values);
    } else {
        static_assert(std::is_same_v<MatrixT, NestedMatrix>,
                      "makeMatrix only supports FlatMatrix and NestedMatrix.");

        std::vector<std::vector<double>> rows(
            static_cast<std::size_t>(rowCount),
            std::vector<double>(static_cast<std::size_t>(colCount)));

        for (int row = 0; row < rowCount; ++row) {
            for (int col = 0; col < colCount; ++col) {
                rows[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] =
                    values[static_cast<std::size_t>(row) * static_cast<std::size_t>(colCount) +
                           static_cast<std::size_t>(col)];
            }
        }

        return NestedMatrix(rows);
    }
}

} // namespace test
