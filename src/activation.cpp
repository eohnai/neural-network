#include "../include/activation.hpp"
#include <cmath>

double Activation::ReLU(double x) {
    return x < 0 ? 0 : x;
}

double Activation::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}
