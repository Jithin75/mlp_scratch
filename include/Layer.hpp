#ifndef LAYER_HPP
#define LAYER_HPP

#include <iostream>
#include <vector>
#include "Neuron.hpp"
using namespace std;

class Layer
{
    public:
        // Constructor
        Layer(int size);
    private:
        int size;
        vector<Neuron *> neurons; 
};

#endif //LAYER_HPP