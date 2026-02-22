#include "matrix.hpp"
#include <iostream>

int main() {
    Matrix m(3, 3);

    std::cout << "Original Matrix (Zeros):\n";
    m.print();

    m.randomise();

    std::cout << "\nRandomised Matrix:\n";
    m.print();

    return 0;
}