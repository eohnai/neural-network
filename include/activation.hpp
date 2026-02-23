#pragma once

class Activation {
public:
    static double ReLU(double x);
    static double ReLUDerivative(double x);

    static double sigmoid(double x);
    static double sigmoidDerivative(double x);
};