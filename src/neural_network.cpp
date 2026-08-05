#include "neural_network.hpp"

#include "activation.hpp"
#include <random>

NeuralNetwork::NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, ActivationType type,
                             std::optional<std::uint64_t> seed)
    : inputNodeCount(inputNodes),
      hiddenNodeCount(hiddenNodes),
      outputNodeCount(outputNodes),
      learningRate(0.01),
      weightsInputToHidden(hiddenNodes, inputNodes),
      weightsHiddenToOutput(outputNodes, hiddenNodes),
      biasHidden(hiddenNodes, 1),
      biasOutput(outputNodes, 1) {

    if (type == ActivationType::RELU) {
        activationFunction = Activation::ReLU;
        activationDerivative = Activation::ReLUDerivative;
    } else if (type == ActivationType::SIGMOID) {
        activationFunction = Activation::sigmoid;
        activationDerivative = Activation::sigmoidDerivative;
    }

    if (seed.has_value()) {
        std::mt19937_64 generator(*seed);

        weightsInputToHidden.randomise(generator);
        weightsHiddenToOutput.randomise(generator);
        biasHidden.randomise(generator);
        biasOutput.randomise(generator);
    } else {
        weightsInputToHidden.randomise();
        weightsHiddenToOutput.randomise();
        biasHidden.randomise();
        biasOutput.randomise();
    }
}

Matrix NeuralNetwork::forwardLayer(const Matrix &layerInput, const Matrix &weights, const Matrix &bias) const {
    Matrix weightedInputs = weights * layerInput;
    Matrix biasedInputs = weightedInputs + bias;
    return biasedInputs.map(activationFunction);
}

void NeuralNetwork::validateInput(const Matrix &input) const {
    if (input.getRows() != inputNodeCount || input.getCols() != 1) {
        throw std::invalid_argument("Input matrix dimensions must match the network's input nodes (N x 1).");
    }
}

void NeuralNetwork::validateTarget(const Matrix &target) const {
    if (target.getRows() != outputNodeCount || target.getCols() != 1) {
        throw std::invalid_argument("Target matrix dimensions must match the network's output nodes (N x 1).");
    }
}

Matrix NeuralNetwork::forward(const Matrix &input) {
    validateInput(input);

    Matrix hiddenActivations = forwardLayer(input, weightsInputToHidden, biasHidden);
    Matrix outputActivations = forwardLayer(hiddenActivations, weightsHiddenToOutput, biasOutput);

    return outputActivations;
}

void NeuralNetwork::train(const Matrix &input, const Matrix &target) {
    validateInput(input);
    validateTarget(target);

    // Forward pass: calculate activations for the hidden and output layers.
    Matrix hiddenActivations = forwardLayer(input, weightsInputToHidden, biasHidden);
    Matrix outputActivations = forwardLayer(hiddenActivations, weightsHiddenToOutput, biasOutput);

    // Backpropagation: a delta is the loss derivative with respect to a layer's
    // pre-activation values. The output error is the squared-loss derivative
    // with respect to the output activations.
    Matrix outputError = outputActivations - target;
    Matrix outputActivationDerivative = outputActivations.map(activationDerivative);
    Matrix outputDelta = outputError.hadamardProduct(outputActivationDerivative);

    Matrix hiddenError = weightsHiddenToOutput.transpose() * outputDelta;
    Matrix hiddenActivationDerivative = hiddenActivations.map(activationDerivative);
    Matrix hiddenDelta = hiddenError.hadamardProduct(hiddenActivationDerivative);

    // Gradients describe how each parameter changes the loss.
    Matrix weightsHiddenToOutputGradient = outputDelta * hiddenActivations.transpose();
    Matrix biasOutputGradient = outputDelta;
    Matrix weightsInputToHiddenGradient = hiddenDelta * input.transpose();
    Matrix biasHiddenGradient = hiddenDelta;

    const auto scaleByLearningRate = [this](double value) {
        return value * learningRate;
    };

    // Gradient descent subtracts a learning-rate-scaled gradient from each parameter.
    weightsHiddenToOutput = weightsHiddenToOutput - weightsHiddenToOutputGradient.map(scaleByLearningRate);
    biasOutput = biasOutput - biasOutputGradient.map(scaleByLearningRate);
    weightsInputToHidden = weightsInputToHidden - weightsInputToHiddenGradient.map(scaleByLearningRate);
    biasHidden = biasHidden - biasHiddenGradient.map(scaleByLearningRate);
}
