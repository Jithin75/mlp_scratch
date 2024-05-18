#include <iostream>
#include <vector>
#include <stdio.h>
#include <fstream>
#include <streambuf>
#include <ostream>
#include <time.h>
#include "../include/NeuralNetwork.hpp"
#include "../include/json.hpp"

using json = nlohmann::json;

void printSyntax() {
    std::cout << "Syntax:" << std::endl;
    std::cout << "ann [configFile]" << std::endl; 
}

int main(int argc, char **argv) {

    if(argc != 2) {
        printSyntax();
        exit(-1);
    }

    std::ifstream configFile(argv[1]);
    std::string str((std::istreambuf_iterator<char>(configFile)), std::istreambuf_iterator<char>());

    auto config = json::parse(str);

    // Define the topology of the neural network
    std::vector<int> topology = config["topology"];
    double learningRate = config["learningRate"];
    double momentum = config["momentum"];
    double bias = config["bias"];

    for(auto &v:topology) {
        std::cout<<v<<std::endl;
    }
    std::cout<<learningRate<<std::endl;
    std::cout<<momentum<<std::endl;
    std::cout<<bias<<std::endl;

    return 0;

    // Create the neural network 
    NeuralNetwork nn(topology, 3, 2, 1, bias, learningRate, momentum);

    // Define the input to the neural network
    std::vector<double> input = {0.2, 0.5, 0.1};

    // Define the target output for training
    std::vector<double> target = {0.1, 0.6, 0.1};

    // Number of epochs to train the neural network
    int epochs = 1000;

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        nn.train(input, target);

        // Print the total error for this epoch
        // std::cout << "Epoch " << epoch + 1 << " error: " << nn.getTotalError() << std::endl;
    }

    // Perform feed forward one last time to get the final output
    nn.feedForward();

    // Print the final network's output
    std::cout << "Final network output:" << std::endl;
    nn.prettyPrintOutput();

    // Print the historical errors
    nn.prettyPrintHistoricalErrors();

    return 0;
}
