#include "../include/Matrix.hpp"

// Random Number generator from -0.0001 to 0.0001 for initial weights
double Matrix::getRandNum() {
    // Create a random number generator engine
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Create a uniform distribution between -0.0001 and 0.0001
    std::uniform_real_distribution<double> dis(-0.0001, 0.0001);
    
    // Generate a random number
    return dis(gen);
}

// Set Functions
void Matrix::setVal(int r, int c, double val) {
    this->values.at(r).at(c) = val;
}

// Constructor
Matrix::Matrix(int totalRows, int totalCols, bool isRandom) {
    this->totalRows = totalRows;
    this->totalCols = totalCols;
    for(int i = 0; i < totalRows; i++) {
        std::vector<double> row_values;
        for(int j = 0; j < totalCols; j++) {
            double r = isRandom ? this->getRandNum() : 0;
            row_values.push_back(r);
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

// Return a copy of a Matrix
Matrix *Matrix::copy() {
    Matrix *m = new Matrix(this->totalCols, this->totalRows, false);
    for(int i = 0; i < this->totalCols; i++) {
        for(int j = 0; j < this->totalRows; j++) {
            m->setVal(i,j,this->getVal(i,j));
        }
    }
    return m;
}

Matrix *Matrix::operator*(const Matrix& mat) const {
    // Check if multiplication is possible
    if (this->totalCols != mat.totalRows) {
        std::cerr << "Error: Incompatible dimensions for matrix multiplication\n";
        assert(false);
    }

    // Create a new matrix to store the result
    Matrix *result = new Matrix(this->totalRows, mat.totalCols, false);

    // Perform matrix multiplication
    for (int i = 0; i < this->totalRows; ++i) {
        for (int j = 0; j < mat.totalCols; ++j) {
            double sum = 0.0;
            for (int k = 0; k < this->totalCols; ++k) {
                sum += this->values[i][k] * mat.values[k][j];
            }
            result->setVal(i, j, sum);
        }
    }

    return result;
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
