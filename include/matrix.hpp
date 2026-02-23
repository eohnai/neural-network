#pragma once

#include <functional>
#include <vector>

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
    Matrix subtract(const Matrix &other) const;
    Matrix dot(const Matrix &other) const;
    Matrix hadamardProduct(const Matrix &other) const;
    Matrix transpose() const;

    void print() const;

    // operator overloading
    Matrix operator+(const Matrix &other) const;
    Matrix operator-(const Matrix &other) const;
    Matrix operator*(const Matrix &other) const;
    double operator()(int row, int col) const; // reading
    double &operator()(int row, int col);      // writing
    bool operator==(const Matrix &other) const;

    // flexible activation / activation derivative function - lambda
    Matrix map(const std::function<double(double)> &func) const;
};