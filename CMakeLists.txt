cmake_minimum_required(VERSION 3.14)
project(NeuralNetwork)
set(CMAKE_CXX_STANDARD 17)

include_directories(include)

# main app setup
add_executable(network
    src/main.cpp
    src/matrix.cpp
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
    tests/add_test.cpp
    tests/dot_test.cpp 
    tests/transpose_test.cpp
    src/matrix.cpp
)

target_link_libraries(run_tests GTest::gtest_main)