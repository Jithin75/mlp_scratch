#include "../include/Layer.hpp"

// Set Functions
void Layer::setVal(int i, double val) {
    this->neurons.at(i)->setVal(val);
}
void Layer::setNeurons(std::vector<Neuron *> n) {
    this->neurons = n;
}

// Get Functions
std::vector<double> Layer::getActivatedVals() {
    std::vector<double> r;
    for(int i = 0; i < this->neurons.size(); i++) {
        r.push_back(this->neurons.at(i)->getActivatedVal());
    }
    return r;
}

//Constructor
Layer::Layer(int size) {
    this->size = size;
    for(int i = 0; i < size; i++) {
        Neuron *n = new Neuron(0.00);
        this->neurons.push_back(n);
    }
}
Layer::Layer(int size, int activationType) {
    this->size = size;
    for(int i = 0; i < size; i++) {
        Neuron *n = new Neuron(0.00, activationType);
        this->neurons.push_back(n);
    }
}

// Copy Functions
void Layer::layerCopy(std::vector<double> values) {
    assert(values.size() == this->neurons.size());
    for(int i = 0; i < values.size(); i++) {
        this->setVal(i, values.at(i));
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
            std::cerr << "Invalid Type (Given Type: "<< type <<") to matrixify a layer!";
            assert(false);
        }
    }
    return m;
}

