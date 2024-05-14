#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include "../include/Matrix.hpp"

// Random Number generator from 0 to 1 for initial weights
float Matrix::getRandNum() {
    // Create a random number generator engine
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Create a uniform distribution between 0 and 1
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    
    // Generate a random number
    return dis(gen);
}

// Set Functions
void Matrix::setVal(int r, int c, float val) {
    this->values.at(r).at(c) = val;
}

// Constructor
Matrix::Matrix(int totalRows, int totalCols, bool isRandom) {
    this->totalRows = totalRows;
    this->totalCols = totalCols;
    for(int i = 0; i < totalRows; i++) {
        std::vector<float> row_values;
        for(int j = 0; j < totalCols; j++) {
            if(isRandom) {
               row_values.push_back(getRandNum()); 
            } else {
                row_values.push_back(0);
            }
        }
        this->values.push_back(row_values);
    }
}

// Transpose the matrix for forward pass calculation
Matrix *Matrix::transpose() {
    Matrix *m = new Matrix(this->totalCols, this->totalRows, false);
    for(int i = 0; i < this->totalCols; i++) {
        for(int j = 0; j < this->totalRows; j++) {
            m->setVal(i,j,this->getVal(j,i));
        }
    }
    return m;
}

// Visualise Matrix
void Matrix::prettyPrintMatrix() {
    // Iterate over each row
    for (int i = 0; i < totalRows; ++i) {
        std::cout << "+";
        // Print horizontal border
        for (int j = 0; j < totalCols; ++j) {
            std::cout << std::setw(9) << std::setfill('-') << "+";
        }
        std::cout << std::endl << "|";

        // Print matrix values
        for (int j = 0; j < totalCols; ++j) {
            std::cout << std::setw(8) << std::setfill(' ') << std::fixed << std::setprecision(4) << values[i][j] << "|";
        }
        std::cout << std::endl;
    }
    
    // Print the bottom border
    std::cout << "+";
    for (int j = 0; j < totalCols; ++j) {
        std::cout << std::setw(9) << std::setfill('-') << "+";
    }
    std::cout << std::endl;
}
