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
    std::function<double(double)> activation;
    std::function<double(double)> activationDerivative;

    // matrices for weights
    Matrix weight_ih;
    Matrix weight_ho;
    Matrix bias_h;
    Matrix bias_o;

    Matrix computeLayer(Matrix &input, Matrix &weights, Matrix &biases, const std::function<double(double)> &func);
    double MSE(double target, double predicted);

public:
    // constructor
    Network(int inputNodes, int hiddenNodes, int outputNodes, ActivationType type);

    // methods
    Matrix forward(Matrix input);
    void train(Matrix input, Matrix target);
};