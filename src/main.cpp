#include <iostream>
#include "../include/Neuron.hpp"
#include "../include/Matrix.hpp"
#include "../include/NeuralNetwork.hpp"

using namespace std;

int main(int argc, char **argv) {

    vector<int> topology = {3,2,1};
    vector<double> input = {1,0,1};

    NeuralNetwork *nn = new NeuralNetwork(topology);
    nn->setInitialInput(input);
    nn->feedForward();
    nn->prettyPrintNetwork();

    return 0;
}