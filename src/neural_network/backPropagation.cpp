#include "../../include/NeuralNetwork.hpp"

// Back Propagation Implementation
void NeuralNetwork::backPropagation() {
    int indexOutputLayer = this->topologySize - 1;
    
    std::unique_ptr<Matrix> gradients;
    std::unique_ptr<Matrix> derivedValues;
    std::unique_ptr<Matrix> zValues;
    std::unique_ptr<Matrix> zValuesTranspose;
    std::unique_ptr<Matrix> deltaWeights;
    std::unique_ptr<Matrix> transposePrevWeights;
    std::unique_ptr<Matrix> hiddenDerived;
    std::unique_ptr<Matrix> prevGradients;
    std::unique_ptr<Matrix> zActivatedVals;
    std::unique_ptr<Matrix> transposeHidden;

    std::vector<std::unique_ptr<Matrix>> newWeights;

    // Output to last hidden layer
    gradients = std::make_unique<Matrix>(1, topology.at(indexOutputLayer), false);

    derivedValues = std::unique_ptr<Matrix>(this->layers.at(indexOutputLayer)->matrixify(2));
    for(int i = 0; i < this->topology.at(indexOutputLayer); i++) {
        gradients->setVal(0, i, (this->derivedErrors.at(i) * derivedValues->getVal(0, i)));
    }

    zValues = std::unique_ptr<Matrix>(this->layers.at(indexOutputLayer - 1)->matrixify(1));
    zValuesTranspose = std::unique_ptr<Matrix>(zValues->transpose());

    deltaWeights = std::unique_ptr<Matrix>((*zValuesTranspose) * (*gradients));

    auto tempNewWeights = std::make_unique<Matrix>(this->topology.at(indexOutputLayer - 1), this->topology.at(indexOutputLayer), false);
    for(int i = 0; i < this->topology.at(indexOutputLayer - 1); i++) {
        for(int j = 0; j < this->topology.at(indexOutputLayer); j++) {
            double newWeightVal = (this->weightMatrices.at(indexOutputLayer - 1)->getVal(i, j) * this->momentum) - (deltaWeights->getVal(i, j) * this->learningRate);
            tempNewWeights->setVal(i, j, newWeightVal);
        }
    }

    newWeights.push_back(std::make_unique<Matrix>(*tempNewWeights));

    // Back Propagation from last hidden layer to input layer
    for(int i = (indexOutputLayer - 1); i > 0; i--) {
        prevGradients = std::move(gradients);

        transposePrevWeights = std::unique_ptr<Matrix>(this->weightMatrices.at(i)->transpose());
        hiddenDerived = std::unique_ptr<Matrix>(this->layers.at(i)->matrixify(2));

        gradients = std::unique_ptr<Matrix>((*prevGradients) * (*transposePrevWeights));
        for(int j = 0; j < hiddenDerived->getRows(); j++) {
            gradients->setVal(0, j, gradients->getVal(0, j) * hiddenDerived->getVal(0, j));
        }

        if(i == 1) {
            zActivatedVals = std::unique_ptr<Matrix>(this->layers.at(0)->matrixify(0));
        } else {
            zActivatedVals = std::unique_ptr<Matrix>(this->layers.at(i - 1)->matrixify(1));
        }

        transposeHidden = std::unique_ptr<Matrix>(zActivatedVals->transpose());
        deltaWeights = std::unique_ptr<Matrix>((*transposeHidden) * (*gradients));

        tempNewWeights = std::make_unique<Matrix>(deltaWeights->getRows(), deltaWeights->getCols(), false);
        for(int r = 0; r < tempNewWeights->getRows(); r++) {
            for(int c = 0; c < tempNewWeights->getCols(); c++) {
                double newWeightVal = (this->weightMatrices.at(i - 1)->getVal(r, c) * this->momentum) - (deltaWeights->getVal(r, c) * this->learningRate);
                tempNewWeights->setVal(r, c, newWeightVal);
            }
        }

        newWeights.push_back(std::make_unique<Matrix>(*tempNewWeights));
    }

    for(auto& matrix : this->weightMatrices) {
        delete matrix;
    }
    this->weightMatrices.clear();

    std::reverse(newWeights.begin(), newWeights.end());
    for(auto& newWeight : newWeights) {
        this->weightMatrices.push_back(new Matrix(*newWeight));
    }
}
