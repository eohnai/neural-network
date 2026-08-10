#include "nested_matrix.hpp"
#include <iostream>
#include <random>
#include <cassert>
#include <stdexcept>

// constructor when provided number of rows and columns
NestedMatrix::NestedMatrix(int rows, int cols) {
    if (rows < 1 || cols < 1) {
        throw std::invalid_argument("Matrix must have at least one row and one column!");
    }

    // assigning dimensions
    this->rows = rows;
    this->cols = cols;

    // create 2D vector and fill it with 0.0
    data = std::vector<std::vector<double>>(rows, std::vector<double>(cols, 0.0));
}

// constructor when provided matrix
NestedMatrix::NestedMatrix(const std::vector<std::vector<double>> &data) {
    if (data.empty()) {
        throw std::invalid_argument("Matrix must have at least one row!");
    }

    if (data[0].empty()) {
        throw std::invalid_argument("Matrix must have at least one column!");
    }

    // check if all rows have the same number of columns
    for (std::size_t row = 1; row < data.size(); ++row) {
        if (data[row].size() != data[0].size()) {
            throw std::invalid_argument("All rows must have the same number of columns!");
        }
    }

    this->data = data;
    this->rows = static_cast<int>(data.size());
    this->cols = static_cast<int>(data[0].size());
}

// randomise weights
void NestedMatrix::randomise() {
    // set up RNG
    thread_local static std::random_device rd;
    thread_local static std::mt19937_64 generator(rd());
    randomise(generator);
}

void NestedMatrix::randomise(std::mt19937_64 &generator) {
    // define distribution range
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i][j] = distribution(generator);
        }
    }
}

// getters
int NestedMatrix::getRows() const {
    return rows;
}

int NestedMatrix::getCols() const {
    return cols;
}

void NestedMatrix::print() const {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << data[i][j] << " ";
        }

        std::cout << "\n";
    }
}

// mathematical operations
NestedMatrix NestedMatrix::add(const NestedMatrix &other) const {
    if (this->rows != other.rows || this->cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for addition!");
    }

    NestedMatrix res(this->rows, this->cols);
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            res.data[i][j] = this->data[i][j] + other.data[i][j];
        }
    }

    return res;
}

NestedMatrix NestedMatrix::subtract(const NestedMatrix &other) const {
    if (this->rows != other.rows || this->cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for subtraction!");
    }

    NestedMatrix res(this->rows, this->cols);
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            res.data[i][j] = this->data[i][j] - other.data[i][j];
        }
    }

    return res;
}

NestedMatrix NestedMatrix::dot(const NestedMatrix &other) const {
    if (this->cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for dot product!");
    }

    NestedMatrix res(this->rows, other.cols);

    /*
    Example:

        a[0][0] * b[0][0] + a[0][1] * b[1][0] = c[0][0];
        a[1][0] * b[0][0] + a[1][1] * b[1][0] = c[1][0];

        ----------

        a[0][0] * b[0][1] + a[0][1] * b[1][1] = c[0][1];
        a[1][0] * b[0][1] + a[1][1] * b[1][1] = c[1][1];
    */

    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < other.cols; ++j) {
            for (int k = 0; k < this->cols; ++k) {
                res.data[i][j] += this->data[i][k] * other.data[k][j];
            }
        }
    }

    return res;
}

NestedMatrix NestedMatrix::hadamardProduct(const NestedMatrix &other) const {
    if (this->rows != other.rows || this->cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for Hadamard Product!");
    }

    NestedMatrix result(this->rows, this->cols);
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            result.data[i][j] = this->data[i][j] * other.data[i][j];
        }
    }

    return result;
}

NestedMatrix NestedMatrix::transpose() const {
    NestedMatrix transposed(this->cols, this->rows);

    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            transposed.data[j][i] = this->data[i][j];
        }
    }

    return transposed;
}

// operator overloadings
NestedMatrix NestedMatrix::operator+(const NestedMatrix &other) const {
    return this->add(other);
}

NestedMatrix NestedMatrix::operator-(const NestedMatrix &other) const {
    return this->subtract(other);
}

NestedMatrix NestedMatrix::operator*(const NestedMatrix &other) const {
    return this->dot(other);
}

// reading
double NestedMatrix::operator()(int row, int col) const {
    return this->data[row][col];
}

// writing
double &NestedMatrix::operator()(int row, int col) {
    assert(row >= 0 && row < rows && col >=0 && col < cols);
    return data[row][col];
}

bool NestedMatrix::operator==(const NestedMatrix &other) const {
    if (this->rows != other.rows || this->cols != other.cols) {
        return false;
    }

    for (int i = 0; i < other.rows; ++i) {
        for (int j = 0; j < other.cols; ++j) {
            if (this->data[i][j] != other.data[i][j])
                return false;
        }
    }

    return true;
}

NestedMatrix NestedMatrix::map(const std::function<double(double)> &func) const {
    NestedMatrix res(this->rows, this->cols);

    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            res.data[i][j] = func(data[i][j]);
        }
    }

    return res;
}
