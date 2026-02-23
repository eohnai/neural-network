#include "network.hpp"
#include "activation.hpp"
#include "matrix.hpp"

Network::Network(int inputNodes, int hiddenNodes, int outputNodes, ActivationType type)
    // initialiser list
    : weight_ih(hiddenNodes, inputNodes),
      weight_ho(outputNodes, hiddenNodes),
      bias_h(hiddenNodes, 1),
      bias_o(outputNodes, 1) {
    this->inputNodes = inputNodes;
    this->hiddenNodes = hiddenNodes;
    this->outputNodes = outputNodes;
    this->learningRate = 0.01;

    if (type == ActivationType::RELU) {
        this->activation = Activation::ReLU;
        this->activationDerivative = Activation::ReLUDerivative;
    } else if (type == ActivationType::SIGMOID) {
        this->activation = Activation::sigmoid;
        this->activationDerivative = Activation::sigmoidDerivative;
    }

    // randomise the weights and biases
    this->weight_ih.randomise();
    this->weight_ho.randomise();
    this->bias_h.randomise();
    this->bias_o.randomise();
}

Matrix Network::computeLayer(const Matrix &input, const Matrix &weights, const Matrix &biases, const std::function<double(double)> &func) {
    // 1. dot product
    Matrix result = weights * input;

    // 2. add biases
    result = result + biases;

    // 3. activation function
    result = result.map(func);

    return result;
}

Matrix Network::forward(const Matrix &input) {
    if (input.getRows() != this->inputNodes || input.getCols() != 1) {
        throw std::invalid_argument("Input matrix dimensions must match the network's input nodes (N x 1).");
    }

    // layer 1: input -> hidden
    Matrix hidden = computeLayer(input, this->weight_ih, this->bias_h, this->activation);

    // layer 2: hidden -> output
    Matrix output = computeLayer(hidden, this->weight_ho, this->bias_o, this->activation);

    return output;
}

void Network::train(const Matrix &input, const Matrix &target) {
    if (input.getRows() != this->inputNodes || input.getCols() != 1) {
        throw std::invalid_argument("Input matrix dimensions must match the network's input nodes (N x 1).");
    }
    
    // 1. FORWARD PASS
    Matrix hidden = computeLayer(input, this->weight_ih, this->bias_h, this->activation);
    Matrix output = computeLayer(hidden, this->weight_ho, this->bias_o, this->activation);

    // 2. OUTPUT LAYER BACKPROP
    // a. calculate the output error (Prediction - Target)
    Matrix output_error = output - target;

    // b. calculate output gradients
    Matrix output_gradients = output_error.hadamardProduct(output.map(this->activationDerivative));

    double learningRate = this->learningRate;
    output_gradients = output_gradients.map([learningRate](double x) { return x * learningRate; });

    // c. calculate hidden layer error before changing weight_ho!
    // this pushes the error backwards to the hidden nodes
    Matrix hidden_error = this->weight_ho.transpose() * output_error;

    // d. calculate weight deltas and update weights & biases
    Matrix weight_ho_deltas = output_gradients * hidden.transpose();
    this->weight_ho = this->weight_ho - weight_ho_deltas;
    this->bias_o = this->bias_o - output_gradients;

    // 3. HIDDEN LAYER BACKPROP
    Matrix hidden_gradients = hidden_error.hadamardProduct(hidden.map(this->activationDerivative));
    hidden_gradients = hidden_gradients.map([learningRate](double x) { return x * learningRate; });

    Matrix weight_ih_deltas = hidden_gradients * input.transpose();
    this->weight_ih = this->weight_ih - weight_ih_deltas;
    this->bias_h = this->bias_h - hidden_gradients;
}