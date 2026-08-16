cmake_minimum_required(VERSION 3.14)
project(NeuralNetwork)
set(CMAKE_CXX_STANDARD 17)

include_directories(include)

set(MATRIX_IMPLEMENTATION "NESTED" CACHE STRING "Matrix implementation: NESTED or FLAT")
set_property(CACHE MATRIX_IMPLEMENTATION PROPERTY STRINGS NESTED FLAT)
string(TOUPPER "${MATRIX_IMPLEMENTATION}" MATRIX_IMPLEMENTATION)

if(MATRIX_IMPLEMENTATION STREQUAL "NESTED")
    set(MATRIX_SOURCE src/nested_matrix.cpp)
    set(MATRIX_DEFINITION NEURAL_NETWORK_USE_NESTED_MATRIX)
elseif(MATRIX_IMPLEMENTATION STREQUAL "FLAT")
    set(MATRIX_SOURCE src/flat_matrix.cpp)
    set(MATRIX_DEFINITION NEURAL_NETWORK_USE_FLAT_MATRIX)
else()
    message(FATAL_ERROR "MATRIX_IMPLEMENTATION must be NESTED or FLAT, not '${MATRIX_IMPLEMENTATION}'.")
endif()

# main app setup
add_executable(network
    src/activation.cpp
    src/idx_reader.cpp
    src/main.cpp
    src/mnist_dataset.cpp
    ${MATRIX_SOURCE}
    src/neural_network.cpp
)

target_compile_definitions(network PRIVATE
    NEURAL_NETWORK_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    ${MATRIX_DEFINITION}
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
    tests/unit/flat_matrix_test.cpp
    tests/unit/hadamard_product_test.cpp
    tests/unit/idx_reader_test.cpp
    tests/unit/matrix_constructor_test.cpp
    tests/unit/mnist_dataset_test.cpp
    tests/unit/subtract_test.cpp
    tests/unit/transpose_test.cpp
    tests/integration/forward_pass_shape_test.cpp
    tests/integration/network_validation_test.cpp
    tests/integration/xor_test.cpp
    src/activation.cpp
    src/idx_reader.cpp
    src/mnist_dataset.cpp
    src/flat_matrix.cpp
    src/nested_matrix.cpp
    src/neural_network.cpp
)

add_executable(run_benchmark
    tests/benchmarks/matrix_benchmark.cpp
    src/activation.cpp
    ${MATRIX_SOURCE}
)

target_compile_definitions(run_benchmark PRIVATE ${MATRIX_DEFINITION})

add_executable(run_training_benchmark
    tests/benchmarks/training_benchmark.cpp
    src/activation.cpp
    ${MATRIX_SOURCE}
    src/neural_network.cpp
)

target_compile_definitions(run_training_benchmark PRIVATE ${MATRIX_DEFINITION})

target_link_libraries(run_tests GTest::gtest_main)
target_include_directories(run_tests PRIVATE tests/test_support)
target_compile_definitions(run_tests PRIVATE ${MATRIX_DEFINITION})
