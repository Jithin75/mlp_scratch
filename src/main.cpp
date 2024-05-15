#include <iostream>
#include "../include/Neuron.hpp"
#include "../include/Matrix.hpp"
#include "../include/NeuralNetwork.hpp"

using namespace std;

int main(int argc, char **argv) {

    vector<int> topology = {3,2,3};
    vector<double> input = {1,0,1};

    NeuralNetwork *nn = new NeuralNetwork(topology);
    nn->setInitialInput(input);
    nn->setTargetOutput(input);

    // Training NN:
    for(int i = 0; i < 20; i++) {
        std::cout << "Epoch " << i + 1 << ":" << std::endl;
        nn->feedForward();
        nn->setErrors();
        std::cout<< "Total Errors: " << nn->getTotalError() << std::endl;
        nn->backPropogation();
    }
    // nn->prettyPrintNetwork();
    return 0;
}