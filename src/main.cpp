#include "../include/NeuralNetwork.hpp"
#include <iostream>
#include <vector>

int main() {
    // Define the topology of the neural network
    std::vector<int> topology = {3, 2, 3};

    // Create the neural network 
    NeuralNetwork nn(topology, 3, 2, 1, 1.0, 0.05, 1.0);

    // Define the input to the neural network
    std::vector<double> input = {0.2, 0.5, 0.1};

    // Set the initial input to the neural network
    nn.setInitialInput(input);

    // Define the target output for training
    std::vector<double> target = {0.1, 0.6, 0.1};

    // Set the target output of the neural network
    nn.setTargetOutput(target);

    // Number of epochs to train the neural network
    int epochs = 1000;

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Perform feed forward
        nn.feedForward();

        // Set the errors based on the target
        nn.setErrors();

        // Perform backpropagation to adjust weights
        nn.backPropagation();

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
