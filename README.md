# mlp_scratch: Neural Network from Scratch

# Neural Network from Scratch

This project implements a fully customizable neural network from scratch in C++, providing flexibility and control over various hyperparameters and network topology. Key features include JSON-based configuration for easy setup, robust error checking for input consistency, and the ability to save and load network weights. The network supports training and testing with CSV files, predicts values within the 0 to 1 range, and includes a progress bar for real-time training updates. This project is ideal for educational purposes and for those looking to understand the inner workings of neural networks without relying on high-level libraries.

## Getting Started

### Prerequisites

- C++ compiler (e.g., g++)
- CMake
- Make

### Getting the Repository

To get started, first clone the repository and navigate to the project directory:

```sh
git clone https://github.com/Jithin75/mlp_scratch.git
cd mlp_scratch
```

### Compiling the Code

To compile the code, run the following commands within the project directory:

```sh
cmake .
make
```

This will create two executables: `train` and `test`

### Configuration Files

The configuration files for training and testing are in JSON format and contain the necessary parameters for the neural network.
An example defintion of the json file used by `./train`:

```json
{
    "topology": [4, 3, 2],
    "learningRate": 0.05,
    "momentum": 1,
    "bias": 1,
    "epoch": 3000,
    "trainingData": "/path/to/sample_data.csv",
    "labelData": "/path/to/sample_label.csv",
    "weightsFile": "/path/to/weights.json",
    "resultsFile": "/path/to/results.csv"
}
```

Each entry in the JSON file is used by the train executable to configure and train the neural network. Below is an explanation of each entry:

- topology (Required): This specifies the structure of the neural network. It is an array where each element represents the number of neurons in that layer. For example, [4, 3, 2] means the network has an input layer with 4 neurons, one hidden layer with 3 neurons, and an output layer with 2 neurons.
- learningRate (Optional): This specifies the learning rate for the neural network training. It should be a double value. The default value is 0.05. If not provided in the JSON file, the default value will be used.
- momentum (Optional): This specifies the momentum factor for the neural network training. It should be a double value. The default value is 1. If not provided in the JSON file, the default value will be used.
- bias (Optional): This specifies the bias value for the neural network training. It should be a double value. The default value is 1. If not provided in the JSON file, the default value will be used.
- epoch (Optional): This specifies the number of training epochs. It should be an integer value. The default value is 100. If not provided in the JSON file, the default value will be used.
- trainingData (Required): This specifies the path to the CSV file containing the training data. Each row in the CSV file should correspond to one training example.
- labelData (Required): This specifies the path to the CSV file containing the labels for the training data. Each row in the CSV file should correspond to the label for the corresponding training example.
- weightsFile (Optional): This specifies the path to the file where the trained weights will be saved. The default value is ./weights.json. If not provided in the JSON file, the default value will be used.

An example defintion of the json file used by `./train`:

```json
{
    "topology": [4, 3, 2],
    "testData": "/path/to/sample_data.csv",
    "weightsFile": "/path/to/weights.json",
    "resultsFile": "/path/to/results.csv"
}

```

Each entry in the JSON file is used by the test executable to configure and use the neural network to get predicted outputs. Below is an explanation of each entry:
- topology (Required): This specifies the structure of the neural network. It is an array where each element represents the number of neurons in that layer. For example, [4, 3, 2] means the network has an input layer with 4 neurons, one hidden layer with 3 neurons, and an output layer with 2 neurons.
- weightsFile (Required): This specifies the path to the file where the trained weights will be loaded from.
- testData (Required): This specifies the path to the CSV file containing the testing data. Each row in the CSV file should correspond to one testing example.
- resultsFile (Optional): This specifies the path to the file where the results/output of the model on the testing data will be saved. The default value is ./results.csv. If not provided in the JSON file, the default value will be used.

Note: Optional entries should not be included in the JSON file if you want to use the default values. Providing an optional entry with an invalid value (e.g., an empty string) will result in an error.

### Data Files

The data files should be in CSV format. An example formats for the above topology in the config files is shown below.
sample_dat.csv:
```csv
0.1,0.2,0.3,0.4
0.5,0.6,0.7,0.8
0.9,0.1,0.2,0.3
0.4,0.5,0.6,0.7
```
sample_label.csv:
```csv
1,0
0,1
1,0
0,1
```

### Running the Code

#### Training
To train the neural network, use the following command:
```sh
./train config/train.json
```
This will read the configuration from config/train.json, train the neural network using the specified training data, and save the trained weights to the specified file.

#### Testing
To test the neural network, use the following command:
```sh
./test config/test.json
```
This will read the configuration from config/test.json, load the trained weights, test the neural network using the specified test data, and save the results to the specified file.

### Error Checks
The code includes various error checks to ensure proper execution:

- Configuration Parameters: Checks if the required parameters are present and correctly formatted in the configuration file. Default values are used for learningRate, momentum, bias, epoch, and resultsFile if not provided.
- CSV Files: Ensures the CSV files are correctly formatted.
- Topology Consistency: Ensures the topology matches the dimensions of the input and output data.
- Weight File Consistency: Validates the weight file to ensure it matches the neural network topology.

### Example Output
The results of the neural network predictions will be saved in the specified resultsFile in CSV format. Each row in the results.csv file corresponds to the output of the neural network for the respective input from the test data.
results.csv:
```csv
0.862712,0.143946
0.0450902,0.947431
0.95182,0.0472587
0.0986935,0.90011
```

### Upcoming Updates
- Split training data into validation and train sets
- Obtain Accuracy scores for metric-based model evaluation

### Acknowledgements
This project uses the [JSON for Modern C++ library](https://github.com/nlohmann/json/tree/develop) by Niels Lohmann.

### Contact
This README includes code excerpts, instructions on how to run the code, and all necessary information based on the provided conversation. Adjust the paths in the JSON files and CSV examples as needed. If there are additional details you'd like to include or modify, please let me know!
