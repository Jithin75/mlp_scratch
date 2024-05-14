#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <iostream>
#include <vector>
#include "Matrix.hpp"
#include "Layer.hpp"

class NeuralNetwork
{
    public:
        // Constructor
        NeuralNetwork(std::vector<int> topology);

        // Set Initial Input to NN
        void setInitialInput(std::vector<float> input);

        // Visualise Neural Network Topology
        void prettyPrintNetwork();
    private:
        int topologySize;
        std::vector<int> topology;
        std::vector<float> input;
        std::vector<Layer *> layers;
        std::vector<Matrix *> weightMatrices;
};

#endif //NEURALNETWORK_HPP