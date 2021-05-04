#include <stdint.h>
#include "matrix.h"
#include "fixed_point_ops.h"
#include "../utils/utils.h"

// Imports when compiling for the MSP430 device
#ifdef IS_MSP
#include <msp430.h>
#include "DSPLib.h"
#endif

#ifndef MATRIX_OPS_GUARD
#define MATRIX_OPS_GUARD

#define VECTOR_COLS 1
#define VECTOR_INDEX(X)    ((X) * VECTOR_COLS)

// Standard matrix operations
Matrix *matrix_add(Matrix *result, Matrix *mat1, Matrix *mat2);
Matrix *matrix_multiply(Matrix *result, Matrix *mat1, Matrix *mat2, uint16_t precision);
Matrix *apply_elementwise(Matrix *result, Matrix *mat, int16_t (*fn)(int16_t, uint16_t), uint16_t precision);
Matrix *matrix_set(Matrix *mat, int16_t value);
Matrix *matrix_replace(Matrix *dst, Matrix *src);
int16_t dot_product(Matrix *vec1, Matrix *vec2, uint16_t precision);
int16_t argmax(Matrix *vec);

// Sparse Matrix Operations
Matrix *sp_matrix_vector_prod(Matrix *result, SparseMatrix *sp, Matrix *dense, uint16_t precision);

// Block Sparse Matrix Operations
Matrix *block_sparse_matrix_vector_prod(Matrix *result, BlockSparseMatrix *blocks, Matrix *dense, uint16_t precision);

#endif
