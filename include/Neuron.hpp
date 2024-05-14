#ifndef NEURON_HPP
#define NEURON_HPP

#include <iostream>

class Neuron
{
    public:
        // Constructor
        Neuron(float val);

        // Fast sigmoid activation function : f(x) = x/(1 + |x|)
        void activate();

        // Reason: The above function has a simple and easy to calculate derivative - f'(x) = f(x)(1- f(x))
        void derive();

        // Get Functions
        float getVal() {return this->val;}
        float getActivatedVal() {return this->activatedVal;}
        float getDerivedVal() {return this->derivedVal;}

        // Set Functions
        void setVal(float val);
    private:
        // Given Value
        float val;
        // Normalised value (or after activation)
        float activatedVal;
        // Value after derivative
        float derivedVal;
};

#endif //NEURON_HPP