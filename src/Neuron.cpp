#include "../include/Neuron.hpp"

// Set Function
void Neuron::setVal(double val) {
    this->val = val;
    activate();
    derive();
}

// Constructor
Neuron::Neuron(double val) {
    this->setVal(val);
}
Neuron::Neuron(double val, int activationType) {
    this->activationType = activationType;
    this->setVal(val);
}

// Activates based on Activation Type
void Neuron::activate() {
    if(this->activationType == TANH) {
        this->activatedVal = tanh(this->val);
    } else if(this->activationType == SIGM) {
        this->activatedVal = (1.0 / (1.0 + exp(-this->val)));
    } else {
        if(this->val > 0) {
            this->activatedVal = this->val;
        } else {
            this->activatedVal = 0;
        }
    }
}

// Derive based on Activation Type
void Neuron::derive() {
    if(this->activationType == TANH) {
        this->derivedVal = (1.0 - pow(this->activatedVal,2));
    } else if(this->activationType == SIGM) {
        this->derivedVal = (this->activatedVal * (1.0 - this->activatedVal));
    } else {
        if(this->val > 0) {
            this->derivedVal = 1;
        } else {
            this->derivedVal = 0;
        }
    }
}
