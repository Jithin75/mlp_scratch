# mlp_scratch
Explore a customizable neural network framework in C++ with features for designing, training, and analyzing models. Visualize loss graphs and conduct data analytics post-training. Dive into machine learning implementation from scratch.

## Installation

```
$ cmake .
$ make
```

## Features:
- The code is set to define neural networks of any topologies where the error cost is simple difference
- The topology needs to be set in the `main.cpp` file as show in the sample topology
- The training with forward and backward propogation is set to work for a singular input, and works well for such cases, however with multiple input the cost function is not robust enough to guarantee accuracy
- The `main.cpp` file has a sample input and training with a sample topology.

## Improvements in work (Coming very soon!!):
- Refactoring code for readability and better memory allocation to avoid leaks within back propagation
- Allow usage of various activation functions
- Improved cost functions
- Allow users to provide datasets to be trained on
- Allow users to save Model/Weights and visualise data through csv files
