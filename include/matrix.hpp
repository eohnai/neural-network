#pragma once

#if defined(NEURAL_NETWORK_USE_NESTED_MATRIX) && defined(NEURAL_NETWORK_USE_FLAT_MATRIX)
#error "Only one matrix implementation can be selected."
#elif defined(NEURAL_NETWORK_USE_NESTED_MATRIX)
#include "nested_matrix.hpp"
using Matrix = NestedMatrix;
inline constexpr const char *matrixImplementationName = "NestedMatrix";
#elif defined(NEURAL_NETWORK_USE_FLAT_MATRIX)
#include "flat_matrix.hpp"
using Matrix = FlatMatrix;
inline constexpr const char *matrixImplementationName = "FlatMatrix";
#else
#error "Select a matrix implementation with CMake: NESTED or FLAT."
#endif
