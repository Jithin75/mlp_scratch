#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include "../include/json.hpp"
#include "../include/NeuralNetwork.hpp"
#include "../include/utils/Misc.hpp"

using json = nlohmann::json;

bool fileExists(const std::string& filePath) {
    std::ifstream file(filePath);
    return file.good();
}

void saveResults(const std::vector<std::vector<double>>& results, const std::string& resultsFile) {
    std::ofstream outFile(resultsFile);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open results file for writing: " << resultsFile << std::endl;
        return;
    }
    for (const auto& result : results) {
        for (size_t i = 0; i < result.size(); ++i) {
            outFile << result[i];
            if (i < result.size() - 1) {
                outFile << ",";
            }
        }
        outFile << "\n";
    }
    outFile.close();
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

bool validateWeights(const std::vector<Matrix*>& weights, const std::vector<int>& topology) {
    if (weights.size() != topology.size() - 1) {
        std::cerr << "Error: Number of weight layers does not match the topology." << std::endl;
        return false;
    }

    for (size_t i = 0; i < weights.size(); ++i) {
        if (weights[i]->getRows() != topology[i]) {
            std::cerr << "Error: Weight matrix row count does not match the topology at layer " << i << "." << std::endl;
            return false;
        }
        if (weights[i]->getCols() != topology[i + 1]) {
            std::cerr << "Error: Weight matrix column count does not match the topology at layer " << i << "." << std::endl;
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

    json config;
    try {
        config = json::parse(str);
    } catch (json::parse_error& e) {
        std::cerr << "Error: Failed to parse config file. " << e.what() << std::endl;
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

    std::string testFile = config.value("testData", "");
    std::string weightsFile = config.value("weightsFile", "");
    std::string resultsFile = config.value("resultsFile", "results.csv");

    // Validate required fields
    if (testFile.empty()) {
        std::cerr << "Error: Test data file not provided in the config file." << std::endl;
        return 1;
    }
    if (!fileExists(testFile)) {
        std::cerr << "Error: Test data file not found: " << testFile << std::endl;
        return 1;
    }
    if (weightsFile.empty()) {
        std::cerr << "Error: Weights file not provided in the config file." << std::endl;
        return 1;
    }
    if (!fileExists(weightsFile)) {
        std::cerr << "Error: Weights file not found: " << weightsFile << std::endl;
        return 1;
    }
    if (topology.size() < 2) {
        std::cerr << "Error: Invalid topology. It must contain at least two layers (input and output layers)." << std::endl;
        return 1;
    }

    // Display the parsed configuration
    std::cout << "Network Topology: ";
    for (const auto& v : topology) {
        std::cout << v << " ";
    }
    std::cout << std::endl;

    std::cout << "Weights File: " << weightsFile << std::endl;
    std::cout << "Test Data File: " << testFile << std::endl;
    std::cout << "Results File: " << resultsFile << std::endl;

    // Load test data
    std::vector<std::vector<double>> testData = utils::Misc::getData(testFile);
    if (!validateCSV(testData)) {
        return 1;
    }

    // Check input-output consistency
    if (!testData.empty() && testData.front().size() != topology.front()) {
        std::cerr << "Error: Input layer size does not match test data dimension." << std::endl;
        return 1;
    }

    // Create and initialize the neural network
    NeuralNetwork nn(topology, 3, 2, 1);
    try {
        nn.loadWeights(weightsFile);
    } catch (const std::exception& e) {
        std::cerr << "Error: Failed to load weights. " << e.what() << std::endl;
        return 1;
    }
    if(!validateWeights(nn.getWeightMatrices(), topology)) {
        return 1;
    }

    std::vector<std::vector<double>> results;
    for (const auto& input : testData) {
        nn.setInitialInput(input);
        nn.feedForward();
        results.push_back(nn.getOutput());
    }

    // Save results to CSV file
    saveResults(results, resultsFile);

    std::cout << "Done! Results saved to " << resultsFile << std::endl;

    return 0;
}
