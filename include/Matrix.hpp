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
        double getRandNum();

        // Transpose the matrix for forward pass calculation
        Matrix *transpose();

        // Matrix Multiplication implementation
        Matrix *operator*(const Matrix& mat) const;

        // Visualise Matrix
        void prettyPrintMatrix();

        // Get Functions
        double getVal(int r, int c) {return this->values.at(r).at(c);}
        int getRows() {return this->totalRows;}
        int getCols() {return this->totalCols;}

        // Set Functions
        void setVal(int r, int c, double val);
    private:
        int totalRows;
        int totalCols;
        std::vector<std::vector<double>> values;
};

#endif //MATRIX_HPP