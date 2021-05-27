#include "layers.h"


Matrix *fully_connected(Matrix *result, Matrix *W, Matrix *b, Matrix *inputs, uint8_t use_activation, uint16_t precision) {
    // Apply the weight matrix
    result = matrix_multiply(result, W, inputs, precision);

    // Add the bias vector (if present)
    if (b != NULL_PTR) {
        result = matrix_add(result, b, result);
    }

    // Apply the activation function (if needed)
    if (use_activation) {
        result = apply_elementwise(result, result, &fp16_relu, precision);
    }

    return result;
}


Matrix *sparse_connected(Matrix *result, SparseMatrix *W, Matrix *b, Matrix *inputs, uint8_t use_activation, uint16_t precision) {
    // Apply the weight matrix
    result = sp_matrix_vector_prod(result, W, inputs, precision);

    // Add the bias vector (if present)
    if (b != NULL_PTR) {
        result = matrix_add(result, b, result);
    }

    // Apply the activation function (if needed)
    if (use_activation) {
        result = apply_elementwise(result, result, &fp16_relu, precision);
    }

    return result;
}


Matrix *block_sparse_connected(Matrix *result, BlockSparseMatrix *W, Matrix *b, Matrix *inputs, uint8_t use_activation, uint16_t precision) {
    // Apply the weight matrix
    result = block_sparse_matrix_vector_prod(result, W, inputs, precision);

    // Add the bias vector (if present)
    if (b != NULL_PTR) {
        result = matrix_add(result, b, result);
    }

    // Apply the activation function (if needed)
    if (use_activation) {
        result = apply_elementwise(result, result, &fp16_relu, precision);
    }

    return result;
}


Matrix *block_diagonal_connected(Matrix *result, BlockSparseMatrix *W, Matrix *b, Matrix *shuffleWeights, uint16_t *shuffleIdx, Matrix *inputs, uint8_t use_activation, uint16_t precision) {
    // Apply the weight matrix
    result = block_sparse_matrix_vector_prod(result, W, inputs, precision);

    // Apply the shuffle weights
    int16_t tempData[result->numRows * result->numCols];
    Matrix temp = { tempData, result->numRows, result->numCols };

    shuffled_vector_hadamard(&temp, result, shuffleWeights, shuffleIdx, precision);
    result = matrix_add(result, &temp, result);

    // Add the bias vector (if present)
    if (b != NULL_PTR) {
        result = matrix_add(result, b, result);
    }

    // Apply the activation function (if needed)
    if (use_activation) {
        result = apply_elementwise(result, result, &fp16_relu, precision);
    }

    return result;
}

