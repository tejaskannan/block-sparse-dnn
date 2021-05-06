#include "main.h"

int main(void) {

    // Create a dummy input
    int16_t inputData[6] = { 0, 1, 2, 3, 4, 5 };
    Matrix inputs = { inputData, 6, VECTOR_COLS };

    int16_t resultData[2];
    Matrix result = { resultData, 2, VECTOR_COLS };

    block_sparse_mlp(&result, &inputs, FIXED_POINT_PRECISION);

    printf("Pred: %d %d\n", result.data[0], result.data[1]);
    return 0;
}
