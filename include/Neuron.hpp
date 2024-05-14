#ifndef NEURON_HPP
#define NEURON_HPP

#include <iostream>

class Neuron
{
    public:
        // Constructor
        Neuron(double val);

        // Fast sigmoid activation function : f(x) = x/(1 + |x|)
        void activate();

        // Reason: The above function has a simple and easy to calculate derivative - f'(x) = f(x)(1- f(x))
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
};

#endif //NEURON_HPP