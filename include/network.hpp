#include "matrix.hpp"
#include <functional>

class Network {
private:
    // 3 layers of nodes
    int inputNodes;
    int hiddenNodes;
    int outputNodes;

    // matrices for weights
    Matrix weight_ih;
    Matrix weight_ho;
    Matrix bias_h;
    Matrix bias_o;

    Matrix computeLayer(Matrix &input, Matrix &weights, Matrix &biases, const std::function<double(double)> &func);

public:
    // constructor
    Network(int inputNodes, int hiddenNodes, int outputNodes);

    // methods
    Matrix forward(Matrix input);
};