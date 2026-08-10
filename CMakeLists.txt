cmake_minimum_required(VERSION 3.14)
project(NeuralNetwork)
set(CMAKE_CXX_STANDARD 17)

include_directories(include)

# main app setup
add_executable(network
    src/activation.cpp
    src/main.cpp
    src/nested_matrix.cpp
    src/neural_network.cpp
)

# Google Test setup
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

# test version of executables
add_executable(run_tests
    tests/unit/activation_test.cpp
    tests/unit/add_test.cpp
    tests/unit/dot_test.cpp
    tests/unit/hadamard_product_test.cpp
    tests/unit/matrix_constructor_test.cpp
    tests/unit/subtract_test.cpp
    tests/unit/transpose_test.cpp
    tests/integration/forward_pass_shape_test.cpp
    tests/integration/network_validation_test.cpp
    tests/integration/xor_test.cpp
    src/activation.cpp
    src/nested_matrix.cpp
    src/neural_network.cpp
)

add_executable(run_benchmark
    tests/benchmarks/matrix_benchmark.cpp
    src/activation.cpp
    src/nested_matrix.cpp
)

add_executable(run_training_benchmark
    tests/benchmarks/training_benchmark.cpp
    src/activation.cpp
    src/nested_matrix.cpp
    src/neural_network.cpp
)

target_link_libraries(run_tests GTest::gtest_main)
