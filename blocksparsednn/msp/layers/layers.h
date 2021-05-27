#include <stdint.h>
#include "../math/matrix.h"
#include "../math/matrix_ops.h"
#include "../math/fixed_point_ops.h"

#ifndef LAYERS_H_
#define LAYERS_H_

    Matrix *fully_connected(Matrix *result, Matrix *W, Matrix *b, Matrix *inputs, uint8_t use_activation, uint16_t precision);
    Matrix *sparse_connected(Matrix *result, SparseMatrix *W, Matrix *b, Matrix *inputs, uint8_t use_activation, uint16_t precision);
    Matrix *block_sparse_connected(Matrix *result, BlockSparseMatrix *W, Matrix *b, Matrix *inputs, uint8_t use_activation, uint16_t precision);
    Matrix *block_diagonal_connected(Matrix *result, BlockSparseMatrix *W, Matrix *b, Matrix *shuffleWeights, uint16_t *shuffleIdx, Matrix *inputs, uint8_t use_activation, uint16_t precision);

#endif
