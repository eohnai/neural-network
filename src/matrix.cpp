#include "matrix.hpp"
#include <iostream>
#include <random>

// constructor
Matrix::Matrix(int rows, int cols) {
    // assigning dimensions
    this->rows = rows;
    this->cols = cols;

    // create 2D vector and fill it with 0.0
    data = std::vector<std::vector<double>>(rows, std::vector<double>(cols, 0.0));
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