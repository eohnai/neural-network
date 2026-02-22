#include "matrix.hpp"

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

public:
};