#include "activation.hpp"
#include <cmath>

double Activation::ReLU(double x) {
    return x <= 0 ? 0 : x;
}

double Activation::ReLUDerivative(double x) {
    return x <= 0 ? 0 : 1;
}

double Activation::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double Activation::sigmoidDerivative(double x) {
    return x * (1 - x);
}