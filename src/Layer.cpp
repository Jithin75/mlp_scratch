#include <iostream>
#include <vector>
#include <cassert>
#include "../include/Layer.hpp"

// Set Functions
void Layer::setNeuronVal(int i, float val) {
    this->neurons.at(i)->setVal(val);
}

//Constructor
Layer::Layer(int size) {
    this->size = size;
    for(int i = 0; i < size; i++) {
        Neuron *n = new Neuron(0.00);
        this->neurons.push_back(n);
    }
}

// Copy Functions
void Layer::layerCopy(std::vector<float> values) {
    assert(values.size() == this->neurons.size());
    for(int i = 0; i < values.size(); i++) {
        this->setNeuronVal(i, values.at(i));
    }
}

// Matrix representation of the Layer values, activation values, and derived values
Matrix *Layer::matrixify(int type) {
    Matrix *m = new Matrix(1,this->size,false);
    for(int i = 0; i < this->size; i++) {
        if(type == 0) {
            m->setVal(0,i,this->neurons.at(i)->getVal());
        } else if (type == 1)
        {
            m->setVal(0,i,this->neurons.at(i)->getActivatedVal());
        } else if (type == 2)
        {
            m->setVal(0,i,this->neurons.at(i)->getDerivedVal());
        } else {
            throw "Invalid Type!";
        }
    }
    return m;
}

