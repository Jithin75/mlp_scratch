#include "../../include/NeuralNetwork.hpp"

// Expected that target is set, and calculates the errors
void NeuralNetwork::setErrors() {
    if (this->target.size() == 0) {
        std::cerr << "No target set for this Neural Network" << std::endl;
        assert(false);
    }

    if (this->target.size() != this->layers.at(this->layers.size() - 1)->getSize()) {
        std::cerr << "Target size (" << this->target.size() <<") do not match the output layer ("<< this->layers.at(this->layers.size() - 1)->getSize()<<") of the Neural Network topology" << std::endl;
        assert(false);
    }

    switch(this->costFunctionType) {
        case (COST_MSE):
            this->setErrorMSE();
            break;
        default:
            this->setErrorMSE();
            break;
    }
}

void NeuralNetwork::setErrorMSE() {
    int outputLayerIndex = this->layers.size() - 1;
    std::vector<Neuron *> outputNeurons = this->layers.at(outputLayerIndex)->getNeurons();

    this->error = 0.0;
    this->errors.clear();
    this->derivedErrors.clear();

    for (size_t i = 0; i < target.size(); ++i) {
        double t = target[i];
        double y = outputNeurons[i]->getActivatedVal();
        double error = 0.5 * pow(t - y, 2);
        this->errors.push_back(error);
        this->derivedErrors.push_back(y - t);
        this->error += error;
    }

    this->historicalErrors.push_back(this->error);
}