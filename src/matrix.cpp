#include "matrix.hpp"
#include <iostream>
#include <random>

// constructor when provided number of rows and columns
Matrix::Matrix(int rows, int cols) {
    // assigning dimensions
    this->rows = rows;
    this->cols = cols;

    // create 2D vector and fill it with 0.0
    data = std::vector<std::vector<double>>(rows, std::vector<double>(cols, 0.0));
}

// constructor when provided matrix
Matrix::Matrix(const std::vector<std::vector<double>> &data) {
    this->data = data;

    // initialise the rows and columns also
    this->rows = data.size();
    this->cols = data[0].size();
}

// randomise weights
void Matrix::randomise() {
    // set up RNG
    std::random_device rd;
    std::mt19937_64 gen(rd());

    // define distribution range
    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i][j] = dis(gen);
        }
    }
}

// getters
int Matrix::getRows() const {
    return rows;
}

int Matrix::getCols() const {
    return cols;
}

void Matrix::print() const {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << data[i][j] << " ";
        }

        std::cout << "\n";
    }
}

// mathematical operations
Matrix Matrix::add(const Matrix &other) const {
    if (this->rows != other.rows || this->cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for addition!");
    }

    Matrix res(this->rows, this->cols);
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            res.data[i][j] = this->data[i][j] + other.data[i][j];
        }
    }

    return res;
}

Matrix Matrix::dot(const Matrix &other) const {
    if (this->cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for dot product!");
    }

    Matrix res(this->rows, other.cols);

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

Matrix Matrix::transpose() const {
    Matrix transposed(this->cols, this->rows);

    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            transposed.data[j][i] = this->data[i][j];
        }
    }

    return transposed;
}

// operator overloadings
double Matrix::operator()(int row, int col) const {
    return this->data[row][col];
}

bool Matrix::operator==(const Matrix &other) const {
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

void Matrix::activation(const std::function<double(double)> &func) {
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            this->data[i][j] = func(data[i][j]);
        }
    }
}