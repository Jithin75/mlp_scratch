#include <iostream>
#include "../include/Neuron.hpp"

using namespace std;

int main(int argc, char **argv) {

    Neuron *n = new Neuron(0.9);
    cout << "Value: " << n->getVal() << endl;
    cout << "Activated Value: " << n->getActivatedVal() << endl;
    cout << "Derived Value: " << n->getDerivedVal() << endl;
    return 0;
}