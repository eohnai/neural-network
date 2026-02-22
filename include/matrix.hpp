#pragma once

#include <vector>

class Matrix {
private:
    // private so that these values can only be changed by matrix methods
    int rows;
    int cols;
    std::vector<std::vector<double>> data;

public:
    // constructor
    Matrix(int rows, int cols);

    // method to randomise weights
    void randomise();

    // getters
    int getRows() const;
    int getCols() const;

    // mathematical functions
    Matrix add(const Matrix &other) const;
    Matrix subtract(const Matrix &other) const;
    Matrix dot(const Matrix &other) const;
    Matrix transpose() const;

    void print() const;
};