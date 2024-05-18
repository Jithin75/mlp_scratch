#ifndef LAYER_HPP
#define LAYER_HPP

#include <iostream>
#include <vector>
#include <cassert>
#include "Neuron.hpp"
#include "Matrix.hpp"

class Layer
{
    public:
        // Constructor
        Layer(int size);
        Layer(int size, int activationType);

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
        void setNeurons(std::vector<Neuron *> n);

        // Get Functions
        int getSize() {return this->size;}
        std::vector<Neuron *> getNeurons() {return this->neurons;}
        std::vector<double> getActivatedVals();
    private:
        int size;
        std::vector<Neuron *> neurons; 
};

#endif //LAYER_HPP