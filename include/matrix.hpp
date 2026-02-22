#pragma once

#include <vector>
#include <functional>

class Matrix {
private:
    // private so that these values can only be changed by matrix methods
    int rows;
    int cols;
    std::vector<std::vector<double>> data;

public:
    // constructors
    Matrix(int rows, int cols);
    Matrix(const std::vector<std::vector<double>> &data);

    // method to randomise weights
    void randomise();

    // getters
    int getRows() const;
    int getCols() const;

    // mathematical functions
    Matrix add(const Matrix &other) const;
    Matrix dot(const Matrix &other) const;
    Matrix transpose() const;

    void print() const;

    // operator overloading
    Matrix operator+() const;
    Matrix operator*() const;
    double operator()(int row, int col) const;
    bool operator==(const Matrix &other) const;

    // flexible activation function - lambda
    void activation(const std::function<double(double)> &func);
};