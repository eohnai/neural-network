#include "flat_matrix.hpp"
#include <cassert>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {

int requirePositiveDimension(int count) {
    if (count < 1) {
        throw std::invalid_argument("Matrix has invalid dimensions!");
    }

    return count;
}

std::size_t elementCount(int rows, int cols) {
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
}

} // namespace

// constructor when provided number of rows and columns only
FlatMatrix::FlatMatrix(int rowCount, int colCount)
    : rows(requirePositiveDimension(rowCount)), cols(requirePositiveDimension(colCount)),
      data(elementCount(rows, cols), 0.0) {}

// constructor when provided number of rows and columns with data
FlatMatrix::FlatMatrix(int rowCount, int colCount, const std::vector<double> &values)
    : rows(requirePositiveDimension(rowCount)), cols(requirePositiveDimension(colCount)),
      data(values) {
    if (data.size() != elementCount(rows, cols)) {
        throw std::invalid_argument("Data size does not match matrix dimensions!");
    }
}

// randomise weights
void FlatMatrix::randomise() {
    // set up RNG
    thread_local static std::random_device rd;
    thread_local static std::mt19937_64 generator(rd());
    randomise(generator);
}

void FlatMatrix::randomise(std::mt19937_64 &generator) {
    // define distribution range
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    for (std::size_t index = 0; index < data.size(); ++index) {
        data[index] = distribution(generator);
    }
}

// getters
int FlatMatrix::getRows() const { return rows; }

int FlatMatrix::getCols() const { return cols; }

void FlatMatrix::print() const {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            std::cout << data[row * cols + col] << " ";
        }

        std::cout << '\n';
    }
}

void FlatMatrix::requireSameShape(const FlatMatrix &other, const std::string &operation) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for " + operation + "!");
    }
}

// mathematical operations
FlatMatrix FlatMatrix::add(const FlatMatrix &other) const {
    this->requireSameShape(other, "addition");

    FlatMatrix result(rows, cols);

    for (std::size_t index = 0; index < data.size(); ++index) {
        result.data[index] = data[index] + other.data[index];
    }

    return result;
}

FlatMatrix FlatMatrix::subtract(const FlatMatrix &other) const {
    this->requireSameShape(other, "subtraction");

    FlatMatrix result(rows, cols);

    for (std::size_t index = 0; index < data.size(); ++index) {
        result.data[index] = data[index] - other.data[index];
    }

    return result;
}

FlatMatrix FlatMatrix::dot(const FlatMatrix &other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for dot product!");
    }

    FlatMatrix result(rows, other.cols);

    /*
    Example:

        a[0][0] * b[0][0] + a[0][1] * b[1][0] = c[0][0];
        a[1][0] * b[0][0] + a[1][1] * b[1][0] = c[1][0];

        ----------

        a[0][0] * b[0][1] + a[0][1] * b[1][1] = c[0][1];
        a[1][0] * b[0][1] + a[1][1] * b[1][1] = c[1][1];
    */

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < other.cols; ++j) {
            for (int k = 0; k < cols; ++k) {
                result.data[i * result.cols + j] +=
                    this->data[i * cols + k] * other.data[k * other.cols + j];
            }
        }
    }

    return result;
}

FlatMatrix FlatMatrix::hadamardProduct(const FlatMatrix &other) const {
    this->requireSameShape(other, "Hadamard product");

    FlatMatrix result(rows, cols);

    for (std::size_t index = 0; index < data.size(); ++index) {
        result.data[index] = data[index] * other.data[index];
    }

    return result;
}

FlatMatrix FlatMatrix::transpose() const {
    FlatMatrix transposed(cols, rows);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed.data[j * rows + i] = this->data[i * cols + j];
        }
    }

    return transposed;
}

// operator overloadings
FlatMatrix FlatMatrix::operator+(const FlatMatrix &other) const { return this->add(other); }

FlatMatrix FlatMatrix::operator-(const FlatMatrix &other) const { return this->subtract(other); }

FlatMatrix FlatMatrix::operator*(const FlatMatrix &other) const { return this->dot(other); }

// reading
double FlatMatrix::operator()(int row, int col) const {
    assert(row >= 0 && row < rows && col >= 0 && col < cols);
    return data[row * cols + col];
}

// writing
double &FlatMatrix::operator()(int row, int col) {
    assert(row >= 0 && row < rows && col >= 0 && col < cols);
    return data[row * cols + col];
}

bool FlatMatrix::operator==(const FlatMatrix &other) const {
    if (rows != other.rows || cols != other.cols) {
        return false;
    }

    return data == other.data;
}

FlatMatrix FlatMatrix::map(const std::function<double(double)> &func) const {
    FlatMatrix res(rows, cols);

    for (std::size_t index = 0; index < data.size(); ++index) {
        res.data[index] = func(data[index]);
    }

    return res;
}
