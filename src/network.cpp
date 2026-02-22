#include "../include/network.hpp"
#include "../include/activation.hpp"
#include "../include/matrix.hpp"

Network::Network(int inputNodes, int hiddenNodes, int outputNodes)
    // initialiser list
    : weight_ih(hiddenNodes, inputNodes),
      weight_ho(outputNodes, hiddenNodes),
      bias_h(hiddenNodes, 1),
      bias_o(outputNodes, 1) {
    this->inputNodes = inputNodes;
    this->hiddenNodes = hiddenNodes;
    this->outputNodes = outputNodes;

    // randomise the weights and biases
    this->weight_ih.randomise();
    this->weight_ho.randomise();
    this->bias_h.randomise();
    this->bias_o.randomise();
}

Matrix Network::computeLayer(Matrix &input, Matrix &weights, Matrix &biases, const std::function<double(double)> &func) {
    // 1. dot product
    Matrix result = weights * input;

    // 2. add biases
    result = result + biases;

    // 3. activation function
    result.activation(func);

    return result;
}

Matrix Network::forward(Matrix input) {
    if (input.getRows() != this->inputNodes || input.getCols() != 1) {
        throw std::invalid_argument("Input matrix dimensions must match the network's input nodes (N x 1).");
    }

    // layer 1: input -> hidden
    Matrix hidden = computeLayer(input, this->weight_ih, this->bias_h, Activation::ReLU);

    // layer 2: hidden -> output
    Matrix output = computeLayer(hidden, this->weight_ho, this->bias_o, Activation::ReLU);

    return output;
}