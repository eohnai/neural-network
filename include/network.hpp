#include "matrix.hpp"
#include <functional>

enum class ActivationType {
    RELU,
    SIGMOID
};

class Network {
private:
    // 3 layers of nodes
    int inputNodes;
    int hiddenNodes;
    int outputNodes;
    double learningRate;
    std::function<double(double)> activation;
    std::function<double(double)> activationDerivative;

    // matrices for weights
    Matrix weight_ih;
    Matrix weight_ho;
    Matrix bias_h;
    Matrix bias_o;

    Matrix computeLayer(const Matrix &input, const Matrix &weights, const Matrix &biases, const std::function<double(double)> &func);

public:
    // constructor
    Network(int inputNodes, int hiddenNodes, int outputNodes, ActivationType type);

    // methods
    Matrix forward(const Matrix &input);
    void train(const Matrix &input, const Matrix &target);
};