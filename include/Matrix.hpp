#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <vector>

class Matrix
{
    public:
        // Constructor
        Matrix(int totalRows, int totalCols, bool isRandom);

        // Random Number generator from 0 to 1 for initial weights
        float getRandNum();

        // Transpose the matrix for forward pass calculation
        Matrix *transpose();

        // Visualise Matrix
        void prettyPrintMatrix();

        // Get Functions
        float getVal(int r, int c) {return this->values.at(r).at(c);}
        int getRows() {return this->totalRows;}
        int getCols() {return this->totalCols;}

        // Set Functions
        void setVal(int r, int c, float val);
    private:
        int totalRows;
        int totalCols;
        std::vector<std::vector<float>> values;
};

#endif //MATRIX_HPP