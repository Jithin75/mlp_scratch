#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#define COST_MSE 1

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <time.h>
#include <vector>
#include <cassert>
#include <cmath>
#include <memory>
#include "Matrix.hpp"
#include "Layer.hpp"
#include "json.hpp"

using json = nlohmann::json;

class NeuralNetwork
{
    public:
        // Constructor
        NeuralNetwork(std::vector<int> topology, double bias = 1, double learningRate = 0.05, double momentum = 1);
        NeuralNetwork(std::vector<int> topology, int hiddenActivationType, int outputActivationType, int costFunctionType, double bias = 1, double learningRate = 0.05, double momentum = 1);

        // Set Initial Input to NN
        void setInitialInput(std::vector<double> input);

        // Set Target Output of NN
        void setTargetOutput(std::vector<double> target);

        // Feed Forward Implementation
        void feedForward();

        // Back Propogation Implementation
        void backPropagation();

        // Train Implementation
        void train(std::vector<double> input, std::vector<double> target);

        // Save Weights
        void saveWeights(std::string file);

        // Load Weights
        void loadWeights(std::string file);

        // Visualise Neural Network Topology
        void prettyPrintNetwork();

        // Visualise Output
        void prettyPrintOutput();

        // Visualise Target
        void prettyPrintTarget();

        // Visualise Errors
        void prettyPrintHistoricalErrors();

        // Expected that target is set, and calculates the errors
        void setErrors();
        void setErrorMSE();

        // Get Functions
        Matrix *getNeuronMatrix(int index) { return this->layers.at(index)->matrixify(0); }
        Matrix *getActivatedNeuronMatrix(int index) { return this->layers.at(index)->matrixify(1); }
        Matrix *getDerivedNeuronMatrix(int index) { return this->layers.at(index)->matrixify(2); }
        Matrix *getWeightMatrix(int index) { return new Matrix(*this->weightMatrices.at(index)); };
        void setNeuronValue(int indexLayer, int indexNeuron, double val) { this->layers.at(indexLayer)->setVal(indexNeuron, val); }
        double getTotalError() {return this->error;}
        std::vector<double> getErrors() {return this->errors;}
        std::vector<double> getActivatedVals(int index) {return this->layers.at(index)->getActivatedVals();}
        std::vector<double> getOutput() {return this->layers.at(this->topologySize - 1)->getActivatedVals();}
        std::vector<Matrix *> getWeightMatrices() {return this->weightMatrices;}
    private:
        int topologySize;
        int hiddenActivationType = RELU;
        int outputActivationType = SIGM;
        int costFunctionType = COST_MSE;

        double error = 0;
        double bias = 1;
        double momentum;
        double learningRate;

        std::vector<int> topology;
        std::vector<double> input;
        std::vector<double> target;
        std::vector<double> errors;
        std::vector<double> derivedErrors;
        std::vector<double> historicalErrors;

        std::vector<Layer *> layers;
        std::vector<Matrix *> weightMatrices;
        std::vector<Matrix *> gradientMatrices;
};

#endif //NEURALNETWORK_HPP