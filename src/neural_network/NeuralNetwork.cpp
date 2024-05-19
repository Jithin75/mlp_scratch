#include "../../include/NeuralNetwork.hpp"

// Save Weights
void NeuralNetwork::saveWeights(const std::string file) {
    json j;
    std::vector<std::vector<std::vector<double>>> weightSet;

    // Collect weights from each weight matrix
    for (const auto& matrix : this->weightMatrices) {
        weightSet.push_back(matrix->getValues());
    }

    j["weights"] = weightSet;

    // Write the JSON object to the specified file with pretty formatting
    std::ofstream outFile(file);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file for saving weights: " << file << std::endl;
        return;
    }

    outFile << std::setw(4) << j << std::endl;
    outFile.close();
}

// Load Weights
void NeuralNetwork::loadWeights(const std::string file) {
    std::ifstream inFile(file);
    if (!inFile.is_open()) {
        std::cerr << "Error: Could not open file for loading weights: " << file << std::endl;
        throw std::runtime_error("Failed to open weights file");
    }

    json jWeights;
    try {
        inFile >> jWeights;
    } catch (const json::parse_error& e) {
        std::cerr << "Error: Failed to parse weights file. " << e.what() << std::endl;
        throw std::runtime_error("Failed to parse weights file");
    }
    inFile.close();

    std::vector<std::vector<std::vector<double>>> tmp = jWeights.at("weights").get<std::vector<std::vector<std::vector<double>>>>();

    // Validate the structure of the loaded weights
    if (tmp.size() != this->weightMatrices.size()) {
        std::cerr << "Error: Number of weight layers does not match the network topology." << std::endl;
        throw std::runtime_error("Weights structure mismatch");
    }

    for (size_t i = 0; i < this->weightMatrices.size(); ++i) {
        for (size_t j = 0; j < tmp[i].size(); ++j) {
            for (size_t k = 0; k < tmp[i][j].size(); ++k) {
                this->weightMatrices[i]->setVal(j, k, tmp[i][j][k]);
            }
        }
    }
}

// Set Initial Input to NN
void NeuralNetwork::setInitialInput(std::vector<double> input) {
    this->input = input;

    this->layers.at(0)->layerCopy(input);
}

// Set Target Output of NN
void NeuralNetwork::setTargetOutput(std::vector<double> target) {
    this->target = target;
}

//Constructor
NeuralNetwork::NeuralNetwork(std::vector<int> topology, double bias, double learningRate, double momentum) {
    this->topologySize = topology.size();
    this->topology = topology;
    this->learningRate =learningRate;
    this->momentum = momentum;
    this->bias = bias;

    for(int i = 0; i < this->topologySize; i++) {
        Layer *l;
        if(i > 0 && i < (topologySize - 1)) {
            l = new Layer(topology.at(i), this->hiddenActivationType);
        } else if(i == (topologySize - 1)) {
            l = new Layer(topology.at(i), this->outputActivationType); 
        } else {
            l = new Layer(topology.at(i));
        }
        this->layers.push_back(l);
    }

    for(int j = 0; j < this->topologySize - 1; j++) {
        Matrix *m = new Matrix(topology.at(j), topology.at(j + 1), true);
        this->weightMatrices.push_back(m);
    }

    this->errors = std::vector<double> (this->topology.at(this->topologySize - 1), 0.00);
    this->derivedErrors = std::vector<double> (this->topology.at(this->topologySize - 1), 0.00);
    this->error = 0.00;
}
NeuralNetwork::NeuralNetwork(std::vector<int> topology, int hiddenActivationType, int outputActivationType, int costFunctionType, double bias, double learningRate, double momentum) {
    this->topologySize = topology.size();
    this->topology = topology;
    this->learningRate = learningRate;
    this->momentum = momentum;
    this->bias = bias;
    
    this->hiddenActivationType = hiddenActivationType;
    this->outputActivationType = outputActivationType;
    this->costFunctionType = costFunctionType;

    for(int i = 0; i < this->topologySize; i++) {
        Layer *l;
        if(i > 0 && i < (topologySize - 1)) {
            l = new Layer(topology.at(i), this->hiddenActivationType);
        } else if(i == (topologySize - 1)) {
            l = new Layer(topology.at(i), this->outputActivationType); 
        } else {
            l = new Layer(topology.at(i));
        }
        this->layers.push_back(l);
    }

    for(int j = 0; j < this->topologySize - 1; j++) {
        Matrix *m = new Matrix(topology.at(j), topology.at(j + 1), true);
        this->weightMatrices.push_back(m);
    }

    this->errors = std::vector<double> (this->topology.at(this->topologySize - 1), 0.00);
    this->derivedErrors = std::vector<double> (this->topology.at(this->topologySize - 1), 0.00);
    this->error = 0.00;
}

// Visualise Neural Network Topology
void NeuralNetwork::prettyPrintNetwork() {
    std::cout << "LAYER 1:" << std::endl;
    this->layers.at(0)->matrixify(0)->prettyPrintMatrix();
    std::cout<< std::endl;

    for(int i = 1; i < this->layers.size(); i++) {
        std::cout << "LAYER " << i + 1 << ":" << std::endl;
        this->layers.at(i)->matrixify(1)->prettyPrintMatrix();
        std::cout<< std::endl;
    }

    for(int j = 0; j < this->weightMatrices.size(); j++) {
        std::cout << "WEIGHT MATRIX " << j + 1 << "(between layers " << j + 1 << " and " << j + 2 << "):" << std::endl;
        this->weightMatrices.at(j)->prettyPrintMatrix();
        std::cout<< std::endl;
    }
}

// Visualise Output
void NeuralNetwork::prettyPrintOutput() {
    this->layers.at(this->layers.size() - 1)->matrixify(1)->prettyPrintMatrix();
}

// Visualise Target
void NeuralNetwork::prettyPrintTarget() {
    Layer *temp = new Layer(this->target.size());
    temp->layerCopy(this->target);
    temp->matrixify(0)->prettyPrintMatrix();
    delete temp;
}

// Helper Function used for Subsampling:
std::vector<double> sampleVector(const std::vector<double>& original) {
    std::vector<double> sampled;

    if (original.size() <= 50) {
        sampled = original; // Return all values if there are 50 or fewer
    } else {
        // Calculate the sampling interval
        double interval = static_cast<double>(original.size()) / 50.0;

        // Sample evenly spaced values
        for (int i = 0; i < 50; ++i) {
            int index = static_cast<int>(i * interval);
            sampled.push_back(original[index]);
        }
    }
    return sampled;
}

// Visualise Errors
void NeuralNetwork::prettyPrintHistoricalErrors() {
    std::cout << "Historical Progression of Errors:" << std::endl;
    std::vector<double> sampled = sampleVector(this->historicalErrors);
    for(auto val : sampled) {
        std::cout << val << std::endl; 
    }
}
