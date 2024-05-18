#include "../../include/NeuralNetwork.hpp"

// Feed Forward Implementation
void NeuralNetwork::feedForward() {
    Matrix *a = NULL; // Matrix of neurons in a layer
    Matrix *b = NULL; // Matrix of weights after the above layer
    Matrix *c = NULL; // Matrix of neurons after the above layer
    for(int i = 0; i < (this->topologySize - 1); i++) {
        if (i == 0) {
            a = this->getNeuronMatrix(0);
        } else{
            a = this->getActivatedNeuronMatrix(i);
        }
        b = this->getWeightMatrix(i);
        c = (*a) * (*b);
        for(int j = 0; j < c->getCols(); j++) {
            this->setNeuronValue(i + 1, j, c->getVal(0, j) + this->bias);
        }
        delete a;
        delete b;
        delete c;
    }
}