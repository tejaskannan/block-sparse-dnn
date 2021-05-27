#include "neural_network.h"

// Temporary buffer to intermediate states
#ifdef IS_MSP
#pragma PERSISTENT(ACTIVATIONS);
#endif
#define HALF_SIZE 256
static int16_t ACTIVATIONS[HALF_SIZE * 2] = { 0 };


#ifdef IS_BLOCK_SPARSE
int16_t block_sparse_mlp(Matrix *inputs, uint16_t precision) {
    // Load into the initial input buffer
    Matrix input0 = { ACTIVATIONS, inputs->numRows, inputs->numCols };
    matrix_replace(&input0, inputs);

    // Apply the first layer
    Matrix hidden0 = { ACTIVATIONS + HALF_SIZE, HIDDEN_0_KERNEL.numRows, VECTOR_COLS };
    block_sparse_connected(&hidden0, &HIDDEN_0_KERNEL, &HIDDEN_0_BIAS, &input0, 1, precision); 

    // Apply the second layer
    Matrix hidden1 = { ACTIVATIONS, HIDDEN_1_KERNEL.numRows, VECTOR_COLS };
    block_sparse_connected(&hidden1, &HIDDEN_1_KERNEL, &HIDDEN_1_BIAS, &hidden0, 1, precision);

    // Apply the third layer
    Matrix hidden2 = { ACTIVATIONS + HALF_SIZE, HIDDEN_2_KERNEL.numRows, VECTOR_COLS };
    block_sparse_connected(&hidden2, &HIDDEN_2_KERNEL, &HIDDEN_2_BIAS, &hidden1, 1, precision);

    // Apply the output layer
    Matrix logits = { ACTIVATIONS, OUTPUT_KERNEL.numRows, VECTOR_COLS };
    fully_connected(&logits, &OUTPUT_KERNEL, &OUTPUT_BIAS, &hidden2, 0, precision);

    return argmax(&logits);
}

int16_t block_diagonal_mlp(Matrix *inputs, uint16_t precision) {
    // Load into the initial input buffer
    Matrix input0 = { ACTIVATIONS, inputs->numRows, inputs->numCols };
    matrix_replace(&input0, inputs);

    // Apply the first layer
    Matrix hidden0 = { ACTIVATIONS + HALF_SIZE, HIDDEN_0_KERNEL.numRows, VECTOR_COLS };
    block_diagonal_connected(&hidden0, &HIDDEN_0_KERNEL, &HIDDEN_0_BIAS, &HIDDEN_0_RANDOM_CONN, HIDDEN_0_RANDOM_IDX, &input0, 1, precision); 

    // Apply the second layer
    Matrix hidden1 = { ACTIVATIONS, HIDDEN_1_KERNEL.numRows, VECTOR_COLS };
    block_diagonal_connected(&hidden1, &HIDDEN_1_KERNEL, &HIDDEN_1_BIAS, &HIDDEN_1_RANDOM_CONN, HIDDEN_0_RANDOM_IDX, &hidden0, 1, precision);

    // Apply the third layer
    Matrix hidden2 = { ACTIVATIONS + HALF_SIZE, HIDDEN_2_KERNEL.numRows, VECTOR_COLS };
    block_diagonal_connected(&hidden2, &HIDDEN_2_KERNEL, &HIDDEN_2_BIAS, &HIDDEN_2_RANDOM_CONN, HIDDEN_2_RANDOM_IDX, &hidden1, 1, precision);

    // Apply the output layer
    Matrix logits = { ACTIVATIONS, OUTPUT_KERNEL.numRows, VECTOR_COLS };
    fully_connected(&logits, &OUTPUT_KERNEL, &OUTPUT_BIAS, &hidden2, 0, precision);

    return argmax(&logits);
}

#endif

#ifdef IS_DENSE
int16_t dense_mlp(Matrix *inputs, uint16_t precision) {
    // Load into the initial input buffer
    Matrix input0 = { ACTIVATIONS, inputs->numRows, inputs->numCols };
    matrix_replace(&input0, inputs);

    // Apply the first layer
    Matrix hidden0 = { ACTIVATIONS + HALF_SIZE, HIDDEN_0_KERNEL.numRows, VECTOR_COLS };
    block_sparse_connected(&hidden0, &HIDDEN_0_KERNEL, &HIDDEN_0_BIAS, &input0, 1, precision); 

    // Apply the second layer
    Matrix hidden1 = { ACTIVATIONS, HIDDEN_1_KERNEL.numRows, VECTOR_COLS };
    block_sparse_connected(&hidden1, &HIDDEN_1_KERNEL, &HIDDEN_1_BIAS, &hidden0, 1, precision);

    // Apply the third layer
    Matrix hidden2 = { ACTIVATIONS + HALF_SIZE, HIDDEN_2_KERNEL.numRows, VECTOR_COLS };
    block_sparse_connected(&hidden2, &HIDDEN_2_KERNEL, &HIDDEN_2_BIAS, &hidden1, 1, precision);

    // Apply the output layer
    Matrix logits = { ACTIVATIONS, OUTPUT_KERNEL.numRows, VECTOR_COLS };
    fully_connected(&logits, &OUTPUT_KERNEL, &OUTPUT_BIAS, &hidden2, 0, precision);

    return argmax(&logits);
}
#endif

#ifdef IS_SPARSE
int16_t sparse_mlp(Matrix *inputs, uint16_t precision) {
    // Load into the initial input buffer
    Matrix input0 = { ACTIVATIONS, inputs->numRows, inputs->numCols };
    matrix_replace(&input0, inputs);

    // Apply the first layer
    Matrix hidden0 = { ACTIVATIONS + HALF_SIZE, HIDDEN_0_KERNEL.numRows, VECTOR_COLS };
    sparse_connected(&hidden0, &HIDDEN_0_KERNEL, &HIDDEN_0_BIAS, &input0, 1, precision); 

    // Apply the second layer
    Matrix hidden1 = { ACTIVATIONS, HIDDEN_1_KERNEL.numRows, VECTOR_COLS };
    sparse_connected(&hidden1, &HIDDEN_1_KERNEL, &HIDDEN_1_BIAS, &hidden0, 1, precision);

    // Apply the third layer
    Matrix hidden2 = { ACTIVATIONS + HALF_SIZE, HIDDEN_2_KERNEL.numRows, VECTOR_COLS };
    sparse_connected(&hidden2, &HIDDEN_2_KERNEL, &HIDDEN_2_BIAS, &hidden1, 1, precision);

    // Apply the output layer
    Matrix logits = { ACTIVATIONS, OUTPUT_KERNEL.numRows, VECTOR_COLS };
    fully_connected(&logits, &OUTPUT_KERNEL, &OUTPUT_BIAS, &hidden2, 0, precision);

    return argmax(&logits);
}
#endif

