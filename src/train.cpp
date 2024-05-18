#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include "../include/NeuralNetwork.hpp"
#include "../include/json.hpp"
#include "../include/utils/Misc.hpp"

using json = nlohmann::json;

bool validateTopology(const std::vector<int>& topology, const std::vector<std::vector<double>>& trainingData, const std::vector<std::vector<double>>& labelData) {
    if (topology.empty() || trainingData.empty() || labelData.empty()) {
        return false;
    }
    
    int inputSize = trainingData[0].size();
    int outputSize = labelData[0].size();

    if (topology.front() != inputSize) {
        std::cerr << "Error: Topology input size does not match training data input size." << std::endl;
        return false;
    }
    
    if (topology.back() != outputSize) {
        std::cerr << "Error: Topology output size does not match label data size." << std::endl;
        return false;
    }

    return true;
}

bool validateCSV(const std::vector<std::vector<double>>& data) {
    if (data.empty()) {
        std::cerr << "Error: CSV file is empty or improperly formatted." << std::endl;
        return false;
    }

    size_t expectedSize = data[0].size();
    for (const auto& row : data) {
        if (row.size() != expectedSize) {
            std::cerr << "Error: Inconsistent number of columns in CSV file." << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Syntax: ./train [configFile]" << std::endl; 
        return 1;
    }

    std::ifstream configFile(argv[1]);
    if (!configFile.is_open()) {
        std::cerr << "Error: Could not open config file." << std::endl;
        return 1;
    }

    std::string str((std::istreambuf_iterator<char>(configFile)), std::istreambuf_iterator<char>());
    configFile.close();

    auto config = json::parse(str, nullptr, false);
    if (config.is_discarded()) {
        std::cerr << "Error: Failed to parse config file." << std::endl;
        return 1;
    }

    std::vector<int> topology = config["topology"];
    double learningRate = config["learningRate"];
    double momentum = config["momentum"];
    double bias = config["bias"];
    std::string trainingFile = config["trainingData"];
    std::string labelsFile = config["labelData"];
    std::string weightsFile = config["weightsFile"];
    int epoch = config["epoch"];

    std::cout << "Network Topology: ";
    for (const auto &v : topology) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
    std::cout << "Learning Rate: " << learningRate << std::endl;
    std::cout << "Momentum: " << momentum << std::endl;
    std::cout << "Bias: " << bias << std::endl;
    std::cout << "Epoch: " << epoch << std::endl;
    std::cout << "Training Data File: " << trainingFile << std::endl;
    std::cout << "Label Data File: " << labelsFile << std::endl;
    std::cout << "Weights File: " << weightsFile << std::endl;

    std::vector<std::vector<double>> trainingData = utils::Misc::getData(trainingFile);
    if (!validateCSV(trainingData)) {
        return 1;
    }

    std::vector<std::vector<double>> labelData = utils::Misc::getData(labelsFile);
    if (!validateCSV(labelData)) {
        return 1;
    }

    if (!validateTopology(topology, trainingData, labelData)) {
        return 1;
    }

    // Create the neural network
    NeuralNetwork nn(topology, 3, 2, 1, bias, learningRate, momentum);

    // Training loop
    for (int counter = 0; counter < epoch; ++counter) {
        for (size_t i = 0; i < trainingData.size(); ++i) {
            std::vector<double> input = trainingData.at(i);
            std::vector<double> target = labelData.at(i);
            nn.train(input, target);
        }
        std::cout << "Epoch " << counter + 1 << " error: " << nn.getTotalError() << std::endl;
    }

    std::cout << "Done! Writing Weights to " << weightsFile << std::endl;
    nn.saveWeights(weightsFile);

    return 0;
}
