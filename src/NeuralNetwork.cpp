#include <iostream>
#include <vector>
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
void NeuralNetwork::setInitialInput(std::vector<float> input) {
    this->input = input;

    this->layers.at(0)->layerCopy(input);
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