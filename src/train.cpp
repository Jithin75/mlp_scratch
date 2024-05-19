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

bool fileExists(const std::string& filePath) {
    std::ifstream file(filePath);
    return file.good();
}

void displayProgressBar(size_t current, size_t total, size_t barWidth = 70) {
    if (current > total) return; // Avoid overflow
    float progress = (float)current / total;
    size_t pos = static_cast<size_t>(barWidth * progress);

    std::cout << "[";
    for (size_t i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
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

    // Error checks and setting default values
    if (!config.contains("topology")) {
        std::cerr << "Error: Topology not provided in the config file." << std::endl;
        return 1;
    }
    std::vector<int> topology;
    try {
        topology = config.at("topology").get<std::vector<int>>();
    } catch (json::type_error& e) {
        std::cerr << "Error: Topology must be an array of integers. " << e.what() << std::endl;
        return 1;
    }

    double learningRate = 0.05;
    try {
        if (config.contains("learningRate")) {
            learningRate = config.at("learningRate").get<double>();
        }
    } catch (json::type_error& e) {
        std::cerr << "Error: Learning rate must be a double. " << e.what() << std::endl;
        return 1;
    }

    double momentum = 1.0;
    try {
        if (config.contains("momentum")) {
            momentum = config.at("momentum").get<double>();
        }
    } catch (json::type_error& e) {
        std::cerr << "Error: Momentum must be a double. " << e.what() << std::endl;
        return 1;
    }

    double bias = 1.0;
    try {
        if (config.contains("bias")) {
            bias = config.at("bias").get<double>();
        }
    } catch (json::type_error& e) {
        std::cerr << "Error: Bias must be a double. " << e.what() << std::endl;
        return 1;
    }

    std::string trainingFile = config.value("trainingData", "");
    std::string labelsFile = config.value("labelData", "");
    std::string weightsFile = config.value("weightsFile", "./mlp_scratch/weights/weights.json");

    int epoch = 100;
    try {
        if (config.contains("epoch")) {
            epoch = config.at("epoch").get<int>();
        }
    } catch (json::type_error& e) {
        std::cerr << "Error: Epoch must be an integer. " << e.what() << std::endl;
        return 1;
    }

    // Validate required fields
    if (trainingFile.empty()) {
        std::cerr << "Error: Training data file not provided in the config file." << std::endl;
        return 1;
    }
    if (labelsFile.empty()) {
        std::cerr << "Error: Label data file not provided in the config file." << std::endl;
        return 1;
    }
    if (topology.size() < 2) {
        std::cerr << "Error: Invalid topology. It must contain at least two layers (input and output layers)." << std::endl;
        return 1;
    }

    // Validate file existence
    if (!fileExists(trainingFile)) {
        std::cerr << "Error: Training data file not found: " << trainingFile << std::endl;
        return 1;
    }
    if (!fileExists(labelsFile)) {
        std::cerr << "Error: Label data file not found: " << labelsFile << std::endl;
        return 1;
    }

    // Validate learning rate, momentum, and bias
    if (learningRate <= 0) {
        std::cerr << "Error: Learning rate must be a positive number." << std::endl;
        return 1;
    }
    if (momentum < 0) {
        std::cerr << "Error: Momentum must be a non-negative number." << std::endl;
        return 1;
    }
    if (bias < 0) {
        std::cerr << "Error: Bias must be a non-negative number." << std::endl;
        return 1;
    }

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
        displayProgressBar(counter + 1, epoch);
        std::cout << "Epoch " << counter + 1 << " error: " << nn.getTotalError() << std::endl;
    }

    std::cout << "Done! Writing Weights to " << weightsFile << std::endl;
    nn.saveWeights(weightsFile);

    return 0;
}
