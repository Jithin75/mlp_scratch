#include <iostream>
#include "../include/Matrix.hpp"

using namespace std;

int main(int argc, char **argv) {

    Matrix *m = new Matrix(3,2,true);
    m->prettyPrintMatrix();

    Matrix *p = m->transpose();
    p->prettyPrintMatrix();

    delete m;
    delete p;
    
    return 0;
}