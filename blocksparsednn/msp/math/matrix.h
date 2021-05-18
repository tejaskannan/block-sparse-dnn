#include <stdint.h>

#ifndef MATRIX_GUARD
    #define MATRIX_GUARD

    #define VECTOR_COLS 1
    #define VECTOR_INDEX(X)    ((X) * VECTOR_COLS)

    typedef int16_t dtype;

    typedef struct {
        dtype *data;
        uint16_t numRows;
        uint16_t numCols;
    } Matrix;

    typedef struct {
        dtype *data;
        uint16_t numRows;
        uint16_t numCols;
        uint16_t *rows;
        uint16_t *cols;
        uint16_t nnz;
    } SparseMatrix;

    typedef struct {
        Matrix **blocks;
        uint16_t numBlocks;
        uint16_t numRows;
        uint16_t numCols;
        uint16_t *rows;
        uint16_t *cols;
    } BlockSparseMatrix;

#endif
