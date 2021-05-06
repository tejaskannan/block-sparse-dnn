#include "neural_network.h"

// Temporary buffer to intermediate states
#ifdef IS_MSP
// DSP Lib Data
#endif
static int16_t ACTIVATIONS[512];


#ifdef IS_BLOCK_SPARSE
Matrix *block_sparse_mlp(Matrix *result, Matrix *inputs, uint16_t precision) {
    // Load into the initial input buffer
    Matrix input0 = { ACTIVATIONS, inputs->numRows, inputs->numCols };
    matrix_replace(&input0, inputs);

    // Apply the first layer
    Matrix hidden0 = { ACTIVATIONS + 256, HIDDEN_0_KERNEL.numRows, VECTOR_COLS };
    block_sparse_connected(&hidden0, &HIDDEN_0_KERNEL, &HIDDEN_0_BIAS, &input0, 1, precision); 

    // Apply the second layer
    Matrix hidden1 = { ACTIVATIONS, HIDDEN_1_KERNEL.numRows, VECTOR_COLS };
    block_sparse_connected(&hidden1, &HIDDEN_1_KERNEL, &HIDDEN_1_BIAS, &hidden0, 1, precision);

    // Apply the third layer
    Matrix hidden2 = { ACTIVATIONS + 256, HIDDEN_2_KERNEL.numRows, VECTOR_COLS };
    block_sparse_connected(&hidden2, &HIDDEN_2_KERNEL, &HIDDEN_2_BIAS, &hidden1, 1, precision);

    // Apply the output layer
    fully_connected(result, &OUTPUT_KERNEL, &OUTPUT_BIAS, &hidden2, 0, precision);

    return result;
}
#endif

