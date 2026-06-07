#include "matrix.hpp"
#include <chrono>
#include <iostream>

int main() {
    std::cout << "--- Initializing 2D Matrix Baseline Arrays ---\n";

    // Experiment A Setup (CPU-bound loop test)
    Matrix dot1(1000, 1000);
    Matrix dot2(1000, 1000);
    dot1.randomise();
    dot2.randomise();

    // Experiment B Setup (Cache-bound memory test)
    Matrix cache1(10000, 10000);
    Matrix cache2(10000, 10000);
    cache1.randomise();
    cache2.randomise();

    std::cout << "Initialization complete. Beginning benchmarks...\n\n";

    // =================================================================
    // TIMING EXPERIMENT A: Matrix Multiplication (O(N^3))
    // =================================================================

    // Warmup pass (dummy run to load instructions into CPU cache)
    Matrix warmupDot = dot1.dot(dot2);

    auto startA = std::chrono::steady_clock::now();

    Matrix resultDot = dot1.dot(dot2); // The real timed operation

    auto endA = std::chrono::steady_clock::now();
    auto elapsedA = std::chrono::duration_cast<std::chrono::milliseconds>(endA - startA);
    std::cout << "Matrix Multiplication (1000x1000) Time: " << elapsedA.count() << " ms\n";

    // =================================================================
    // TIMING EXPERIMENT B: Element-Wise Addition (O(N^2) Cache Stress)
    // =================================================================

    // Warmup pass
    Matrix warmupAdd = cache1.add(cache2);

    auto startB = std::chrono::steady_clock::now();

    Matrix resultAdd = cache1.add(cache2); // The real timed operation

    auto endB = std::chrono::steady_clock::now();
    auto elapsedB = std::chrono::duration_cast<std::chrono::milliseconds>(endB - startB);
    std::cout << "Matrix Addition (10000x10000) Time: " << elapsedB.count() << " ms\n\n";

    // =================================================================
    // THE ANTI-CHEAT CHECKSUM
    // =================================================================
    // We print a quick element from each matrix. This forces a side-effect,
    // proving to the compiler that these results are read, preventing it from
    // optimizing away our entire benchmark program in Release mode!
    std::cout << "Verification Checksum (Ensuring loops executed):\n";
    std::cout << "Dot verification slot (0,0): " << resultDot(0, 0) << "\n";
    std::cout << "Add verification slot (0,0): " << resultAdd(0, 0) << "\n";

    return 0;
}