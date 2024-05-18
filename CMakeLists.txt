cmake_minimum_required(VERSION 3.7)
project(mlp_scratch)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

add_executable(
    mlp_scratch
    src/main.cpp
    src/Neuron.cpp
    src/Layer.cpp
    src/Matrix.cpp
    src/neural_network/NeuralNetwork.cpp
    src/neural_network/setErrors.cpp
    src/neural_network/feedForward.cpp
    src/neural_network/backPropagation.cpp
    src/neural_network/train.cpp
)