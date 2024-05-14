#ifndef LAYER_HPP
#define LAYER_HPP

#include <iostream>
#include <vector>
#include "Neuron.hpp"
#include "Matrix.hpp"

class Layer
{
    public:
        // Constructor
        Layer(int size);

        // Matrix representation of the Layer values, activation values, and derived values
        // Types:
        //  0 -> Neuron Values
        //  1 -> Neuron Activation Values
        //  2 -> Neuron Derived Values
        Matrix *matrixify(int type);
        
        // Copy Functions
        void layerCopy(std::vector<double> values);

        // Set Functions
        void setVal(int i, double val);
    private:
        int size;
        std::vector<Neuron *> neurons; 
};

#endif //LAYER_HPP