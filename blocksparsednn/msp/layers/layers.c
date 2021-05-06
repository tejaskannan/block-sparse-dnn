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
        result = apply_elementwise(result, result, &fp16_leaky_relu, precision);
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
        result = apply_elementwise(result, result, &fp16_leaky_relu, precision);
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
        result = apply_elementwise(result, result, &fp16_leaky_relu, precision);
    }

    return result;
}
