#include <cstdint>
#include <functional>
#include <optional>

#include "matrix.hpp"

enum class ActivationType {
    RELU,
    SIGMOID
};

class NeuralNetwork {
private:
    int inputNodeCount;
    int hiddenNodeCount;
    int outputNodeCount;
    double learningRate;
    std::function<double(double)> activationFunction;
    std::function<double(double)> activationDerivative;

    Matrix weightsInputToHidden;
    Matrix weightsHiddenToOutput;
    Matrix biasHidden;
    Matrix biasOutput;

    Matrix forwardLayer(const Matrix &layerInput, const Matrix &weights, const Matrix &bias) const;
    void validateInput(const Matrix &input) const;
    void validateTarget(const Matrix &target) const;

public:
    // constructor
    NeuralNetwork(
        int inputNodes,
        int hiddenNodes,
        int outputNodes,
        ActivationType type,
        std::optional<std::uint64_t> seed = std::nullopt
    );

    // methods
    Matrix forward(const Matrix &input);
    void train(const Matrix &input, const Matrix &target);
};
