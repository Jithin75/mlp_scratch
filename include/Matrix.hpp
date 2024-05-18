#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cassert>
class Matrix
{
    public:
        // Constructor
        Matrix(int totalRows, int totalCols, bool isRandom);

        // Transpose the matrix for forward pass calculation
        Matrix *transpose();

        // Return a copy of a Matrix
        Matrix *copy();

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

        // Random Number generator from 0 to 1 for initial weights
        double getRandNum();
};

#endif //MATRIX_HPP