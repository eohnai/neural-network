#pragma once

#include <functional>
#include <random>
#include <string>
#include <vector>

class FlatMatrix {
  private:
    // private so that these values can only be changed by matrix methods
    int rows;
    int cols;
    std::vector<double> data;

    void requireSameShape(const FlatMatrix &other, const std::string &operation) const;

  public:
    FlatMatrix(int rowCount, int colCount);
    FlatMatrix(int rowCount, int colCount, const std::vector<double> &values);

    // method to randomise weights
    void randomise();
    void randomise(std::mt19937_64 &generator);

    // getters
    int getRows() const;
    int getCols() const;

    // mathematical functions
    FlatMatrix add(const FlatMatrix &other) const;
    FlatMatrix subtract(const FlatMatrix &other) const;
    FlatMatrix dot(const FlatMatrix &other) const;
    FlatMatrix hadamardProduct(const FlatMatrix &other) const;
    FlatMatrix transpose() const;

    void print() const;

    // operator overloading
    FlatMatrix operator+(const FlatMatrix &other) const;
    FlatMatrix operator-(const FlatMatrix &other) const;
    FlatMatrix operator*(const FlatMatrix &other) const;
    double operator()(int row, int col) const; // reading
    double &operator()(int row, int col);      // writing
    bool operator==(const FlatMatrix &other) const;

    // flexible activation / activation derivative function - lambda
    FlatMatrix map(const std::function<double(double)> &func) const;
};
