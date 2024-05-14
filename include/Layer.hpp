#ifndef LAYER_HPP
#define LAYER_HPP

#include <iostream>
#include <vector>
#include "Neuron.hpp"

class Layer
{
    public:
        // Constructor
        Layer(int size);
    private:
        int size;
        std::vector<Neuron *> neurons; 
};

#endif //LAYER_HPP