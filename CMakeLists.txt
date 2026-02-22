cmake_minimum_required(VERSION 3.10)
project(cpp-nn)
set(CMAKE_CXX_STANDARD 17)
include_directories(include)
add_executable(network
    src/main.cpp
    src/matrix.cpp
)