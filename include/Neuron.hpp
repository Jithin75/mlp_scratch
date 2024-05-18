#ifndef NEURON_HPP
#define NEURON_HPP

#define TANH 1
#define SIGM 2
#define RELU 3

#include <iostream>
#include <math.h>

class Neuron
{
    public:
        // Constructor
        Neuron(double val);
        Neuron(double val, int activationType);

        // Activates based on Activation Type
        void activate();

        // Derive based on Activation Type
        void derive();

        // Get Functions
        double getVal() {return this->val;}
        double getActivatedVal() {return this->activatedVal;}
        double getDerivedVal() {return this->derivedVal;}

        // Set Functions
        void setVal(double val);
    private:
        // Given Value
        double val;
        // Normalised value (or after activation)
        double activatedVal;
        // Value after derivative
        double derivedVal;
        // Activation Function
        int activationType = 3;
};

#endif //NEURON_HPP