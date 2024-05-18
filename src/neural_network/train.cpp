#include "../../include/NeuralNetwork.hpp"

void NeuralNetwork::train(std::vector<double> input, std::vector<double> target) {
    this->setInitialInput(input);
    this->setTargetOutput(target);
    this->feedForward();
    this->setErrors();
    this->backPropagation();
}
