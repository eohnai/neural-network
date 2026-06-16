#pragma once

#include <functional>
#include <vector>

class NestedMatrix {
private:
    // private so that these values can only be changed by matrix methods
    int rows;
    int cols;
    std::vector<std::vector<double>> data;

public:
    // constructors
    NestedMatrix(int rows, int cols);
    NestedMatrix(const std::vector<std::vector<double>> &data);

    // method to randomise weights
    void randomise();

    // getters
    int getRows() const;
    int getCols() const;

    // mathematical functions
    NestedMatrix add(const NestedMatrix &other) const;
    NestedMatrix subtract(const NestedMatrix &other) const;
    NestedMatrix dot(const NestedMatrix &other) const;
    NestedMatrix hadamardProduct(const NestedMatrix &other) const;
    NestedMatrix transpose() const;

    void print() const;

    // operator overloading
    NestedMatrix operator+(const NestedMatrix &other) const;
    NestedMatrix operator-(const NestedMatrix &other) const;
    NestedMatrix operator*(const NestedMatrix &other) const;
    double operator()(int row, int col) const; // reading
    double &operator()(int row, int col);      // writing
    bool operator==(const NestedMatrix &other) const;

    // flexible activation / activation derivative function - lambda
    NestedMatrix map(const std::function<double(double)> &func) const;
};
