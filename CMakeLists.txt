cmake_minimum_required(VERSION 3.7)
project(mlp_scratch)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pthread -std=c++14")

add_executable(
    mlp_scratch
    src/main.cpp
    src/Neuron.cpp
)