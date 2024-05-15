#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../include/NeuralNetwork.hpp"

//Constructor
NeuralNetwork::NeuralNetwork(std::vector<int> topology) {
    this->topologySize = topology.size();
    this->topology = topology;

    for(int i = 0; i < this->topologySize; i++) {
        Layer *l = new Layer(topology.at(i));
        this->layers.push_back(l);
    }

    for(int j = 0; j < this->topologySize - 1; j++) {
        Matrix *m = new Matrix(topology.at(j), topology.at(j + 1), true);
        this->weightMatrices.push_back(m);
    }
}

// Set Initial Input to NN
void NeuralNetwork::setInitialInput(std::vector<double> input) {
    this->input = input;

    this->layers.at(0)->layerCopy(input);
}

// Set Target Output of NN
void NeuralNetwork::setTargetOutput(std::vector<double> target) {
    this->target = target;
    this->errors = std::vector<double> (target.size(), 0);
}

// Visualise Neural Network Topology
void NeuralNetwork::prettyPrintNetwork() {
    std::cout << "LAYER 1:" << std::endl;
    this->layers.at(0)->matrixify(0)->prettyPrintMatrix();
    std::cout<< std::endl;

    for(int i = 1; i < this->layers.size(); i++) {
        std::cout << "LAYER " << i + 1 << ":" << std::endl;
        this->layers.at(i)->matrixify(1)->prettyPrintMatrix();
        std::cout<< std::endl;
    }

    for(int j = 0; j < this->weightMatrices.size(); j++) {
        std::cout << "WEIGHT MATRIX " << j + 1 << "(between layers " << j + 1 << " and " << j + 2 << "):" << std::endl;
        this->weightMatrices.at(j)->prettyPrintMatrix();
        std::cout<< std::endl;
    }
}

// Feed Forward Implementation
void NeuralNetwork::feedForward() {
    for(int i = 0; i < (this->layers.size() - 1); i++) {
        Matrix *a = NULL;
        if (i == 0) {
            a = this->getNeuronMatrix(0);
        } else{
            a = this->getActivatedNeuronMatrix(i);
        }
        Matrix *b = this->getWeightMatrix(i);
        Matrix *c = (*a) * (*b);
        for(int j = 0; j < c->getCols(); j++) {
            this->setNeuronValue(i + 1, j, c->getVal(0, j));
        }
        delete a;
        delete b;
        delete c;
    }
}

// Expected that target is set, and calculates the errors
void NeuralNetwork::setErrors() {
    if (this->target.size() == 0) {
        std::cerr << "No target set for this Neural Network" << std::endl;
        assert(false);
    }

    if (this->target.size() != this->layers.at(this->layers.size() - 1)->getSize()) {
        std::cerr << "Target size do not match the last layer of the Neural Network topology" << std::endl;
        assert(false);
    }

    this->error = 0.00;
    std::vector<Neuron *> outputNeurons = this->layers.at(this->layers.size() - 1)->getNeurons();
    for(int i = 0; i < this->target.size(); i++) {
        double temp = pow((outputNeurons.at(i)->getActivatedVal() - target.at(i)), 2);
        this->errors.at(i) = temp;
        this->error += temp;
    }
    this->historicalErrors.push_back(this->error);
}

// Back Propogation Implementation
void NeuralNetwork::backPropogation() {
    std::vector<Matrix *> newWeights;
    Matrix *gradient;

    // Final Output layer to the last hidden layer
    int ouputLayerIndex = this->layers.size() - 1;
    Matrix *derrivedOutputLayer = this->layers.at(ouputLayerIndex)->matrixify(2);
    Matrix *gradientOutputLayer = new Matrix(1, this->layers.at(ouputLayerIndex)->getSize(), false);
    for(int i = 0; i < this->errors.size(); i++) {
        gradientOutputLayer->setVal(0,i,(derrivedOutputLayer->getVal(0,i) * this->errors.at(i)));
    }

    int lastHiddenLayerIndex = ouputLayerIndex - 1;
    Layer *lastHiddenLayer = this->layers.at(lastHiddenLayerIndex);
    Matrix *weightsBeforeOutput = this->weightMatrices.at(lastHiddenLayerIndex);
    Matrix *deltaBeforeOutput = (*lastHiddenLayer->matrixify(1)->transpose()) * (*gradientOutputLayer); 
    
    Matrix *newBeforeOutput = new Matrix(weightsBeforeOutput->getRows(), weightsBeforeOutput->getCols(), false);
    for(int i = 0; i < weightsBeforeOutput->getRows(); i++) {
        for(int j = 0; j < weightsBeforeOutput->getCols(); j++) {
            newBeforeOutput->setVal(i, j, weightsBeforeOutput->getVal(i,j) - deltaBeforeOutput->getVal(i,j));
        }
    }

    newWeights.push_back(newBeforeOutput);
    gradient = new Matrix(gradientOutputLayer->getRows(),gradientOutputLayer->getCols(),false);
    for(int i = 0; i < gradientOutputLayer->getRows(); i++) {
        for(int j = 0; j < gradientOutputLayer->getCols(); j++) {
            gradient->setVal(i, j, gradientOutputLayer->getVal(i,j));
        }
    }

    // Moving backwards from the last hidden layer to the input layer
    for(int i = lastHiddenLayerIndex; i > 0; i--) {
        Layer *curLayer = this->layers.at(i);
        Matrix *deriveCurLayer = curLayer->matrixify(2);
        Matrix *activateCurLayer = curLayer->matrixify(1);
        Matrix *derivedGradients = new Matrix(1, curLayer->getSize(), false);

        Matrix *rightWeightMatrix = this->weightMatrices.at(i);
        Matrix *leftWeightMatrix = this->weightMatrices.at(i - 1);

        for(int i = 0; i < rightWeightMatrix->getRows(); i++) {
            double sum = 0;
            for(int j = 0; j < rightWeightMatrix->getCols(); j++) {
                double p = gradient->getVal(0,j) * rightWeightMatrix->getVal(i,j);
                sum += p;
            }
            derivedGradients->setVal(0, i, sum * activateCurLayer->getVal(0,i));
        }

        Matrix *leftNeurons = (i - 1) == 0 ? this->layers.at(0)->matrixify(0) : this->layers.at(i - 1)->matrixify(1);
        Matrix *deltaWeights = (*leftNeurons->transpose()) * (*derivedGradients); 
        Matrix *newCurWeights = new Matrix(deltaWeights->getRows(), deltaWeights->getCols(), false);
        for(int i = 0; i < deltaWeights->getRows(); i++) {
            for(int j = 0; j < deltaWeights->getCols(); j++) {
                newCurWeights->setVal(i, j, leftWeightMatrix->getVal(i,j) - deltaWeights->getVal(i,j));
            }
        }

        gradient = new Matrix(derivedGradients->getRows(),derivedGradients->getCols(),false);
        for(int i = 0; i < derivedGradients->getRows(); i++) {
            for(int j = 0; j < derivedGradients->getCols(); j++) {
                gradient->setVal(i, j, derivedGradients->getVal(i,j));
            }
        }
        newWeights.push_back(newCurWeights);
    }

    for (int i = newWeights.size() - 1; i >= 0; i--) {
        this->weightMatrices[newWeights.size() - 1 - i] = newWeights[i];
    }
}
