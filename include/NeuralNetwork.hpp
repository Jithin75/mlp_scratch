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
        void setInitialInput(std::vector<double> input);

        // Feed Forward Implementation
        void feedForward();

        // Visualise Neural Network Topology
        void prettyPrintNetwork();

        // Get Functions
        Matrix *getNeuronMatrix(int index) { return this->layers.at(index)->matrixify(0); }
        Matrix *getActivatedNeuronMatrix(int index) { return this->layers.at(index)->matrixify(1); }
        Matrix *getDerivedNeuronMatrix(int index) { return this->layers.at(index)->matrixify(2); }
        Matrix *getWeightMatrix(int index) { return new Matrix(*this->weightMatrices.at(index)); };
        void setNeuronValue(int indexLayer, int indexNeuron, double val) { this->layers.at(indexLayer)->setVal(indexNeuron, val); }
    private:
        int topologySize;
        std::vector<int> topology;
        std::vector<double> input;
        std::vector<Layer *> layers;
        std::vector<Matrix *> weightMatrices;
};

#endif //NEURALNETWORK_HPP