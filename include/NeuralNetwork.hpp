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

        // Set Target Output of NN
        void setTargetOutput(std::vector<double> target);

        // Feed Forward Implementation
        void feedForward();

        // Back Propogation Implementation
        void backPropogation();

        // Visualise Neural Network Topology
        void prettyPrintNetwork();

        // Expected that target is set, and calculates the errors
        void setErrors();

        // Get Functions
        Matrix *getNeuronMatrix(int index) { return this->layers.at(index)->matrixify(0); }
        Matrix *getActivatedNeuronMatrix(int index) { return this->layers.at(index)->matrixify(1); }
        Matrix *getDerivedNeuronMatrix(int index) { return this->layers.at(index)->matrixify(2); }
        Matrix *getWeightMatrix(int index) { return new Matrix(*this->weightMatrices.at(index)); };
        void setNeuronValue(int indexLayer, int indexNeuron, double val) { this->layers.at(indexLayer)->setVal(indexNeuron, val); }
        double getTotalError() {return this->error;}
        std::vector<double> getErrors() {return this->errors;}
    private:
        int topologySize;
        double error;
        std::vector<int> topology;
        std::vector<double> input;
        std::vector<double> target;
        std::vector<double> errors;
        std::vector<double> historicalErrors;
        std::vector<Layer *> layers;
        std::vector<Matrix *> weightMatrices;
        std::vector<Matrix *> gradientMatrices;
};

#endif //NEURALNETWORK_HPP