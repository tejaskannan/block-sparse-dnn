#include "matrix_ops.h"

#ifdef IS_MSP
#include "DSPLib.h"

// For MSP implementations, we allocate memory in the LEA RAM.
// This memory is used when executing Matrix multiplications.
DSPLIB_DATA(MULTIPLY_BUFFER, 4);
static dtype MULTIPLY_BUFFER[1800];

dtype *dma_load(dtype *result, dtype *data, uint16_t n) {
    /**
     * Loads the first n elements of the data array into the result array using
     * DMA.
     */
    // Configure DMA channel 0
    __data20_write_long((uintptr_t) &DMA0SA, (uintptr_t) data);   // Source block address
    __data20_write_long((uintptr_t) &DMA0DA, (uintptr_t) result); // Destination single address
    DMA0SZ = n;                                      // Block size
    DMA0CTL = DMADT_5 | DMASRCINCR_3 | DMADSTINCR_3; // Rpt, inc
    DMA0CTL |= DMAEN;                                // Enable DMA0
    DMA0CTL |= DMAREQ;

    return result;
}
#endif

Matrix *matrix_add(Matrix *result, Matrix *mat1, Matrix *mat2) {
    /**
     * Adds two matrices together elementwise.
     * Result stored stored in the result argument (and returned for convenience).
     */
    // Validate dimensions
    if ((mat1->numRows != mat2->numRows) || (result->numRows != mat1->numRows)) {
        return NULL_PTR;
    }
    
    uint16_t rows = mat1->numRows;
    uint16_t cols = mat1->numCols > mat2->numCols ? mat2->numCols : mat1->numCols;

    uint16_t mat1Offset, mat2Offset, resultOffset;
    uint16_t rowOffset, colOffset;

    // Compute the element-wise sum. We generally add small vectors together, and the LEA
    // provides little (or negative) benefit due to added overhead.
    uint16_t i, j;
    for (i = rows; i > 0; i--) {

        rowOffset = i - 1;

        mat1Offset = rowOffset * mat1->numCols;
        mat2Offset = rowOffset * mat2->numCols;
        resultOffset = rowOffset * result->numCols;

        for (j = cols; j > 0; j--) {
            colOffset = j - 1;
            result->data[resultOffset + colOffset] = fp16_add(mat1->data[mat1Offset + colOffset], mat2->data[mat2Offset + colOffset]);
        }
    }

    return result;
}


Matrix *matrix_multiply(Matrix *result, Matrix *mat1, Matrix *mat2, uint16_t precision) {
    /**
     * Performs Matrix multiplication and stores value in given result array. The implementation
     * depends on whether or not we are compiling for the MSP430 device.
     */

    // Validate dimensions
    if ((mat1->numCols != mat2->numRows) || (mat1->numRows != result->numRows) || (mat2->numCols != result->numCols)) {
        return NULL_PTR;
    }

    // The result will be a [n, p] Matrix
    uint16_t n = mat1->numRows;
    uint16_t m = mat1->numCols;
    uint16_t p = mat2->numCols;

    #ifdef IS_MSP
    // We first transfer the input matrices to the LEA RAM segment. We make this
    // copy efficient using DMA.
    uint16_t offset = 0;
    dtype *mat1Data = dma_load(MULTIPLY_BUFFER, mat1->data, n * m);
    offset += n * m;

    dtype *mat2Data = dma_load(MULTIPLY_BUFFER + offset, mat2->data, m * p);
    offset += m * p;

    dtype *resultData = MULTIPLY_BUFFER + offset;  // Temporary buffer (in LEA RAM) for the result

    // When using the MSP430, we use the LEA for Matrix multiplications. Based on profiling,
    // the LEA can take up to 5x fewer compute cycles than a standard implementation.
    msp_status status;
    msp_matrix_mpy_q15_params mulParams;

    // Initialze LEA metadata
    mulParams.srcARows = n;
    mulParams.srcACols = m;
    mulParams.srcBRows = m;
    mulParams.srcBCols = p;

    // Perform Matrix multiplication using the LEA
    status = msp_matrix_mpy_q15(&mulParams, mat1Data, mat2Data, resultData);
    msp_checkStatus(status);

    // Convert back to the original fixed-point precision. The LEA assumes 15 fractional bits.
    msp_matrix_shift_q15_params shiftParams;
    shiftParams.rows = n;
    shiftParams.cols = p;
    shiftParams.shift = 15 - precision;

    // Perform element-wise shift using the LEA
    if (shiftParams.shift > 0) {
        status = msp_matrix_shift_q15(&shiftParams, resultData, resultData);
        msp_checkStatus(status);
    }

    // Load result back into the given result Matrix
    dma_load(result->data, resultData, n * p);

    #else

    uint16_t i, j, k;
    uint16_t outerRow, innerRow, resultRow;
    int16_t sum, prod;

    for (i = n; i > 0; i--) {
        outerRow = (i - 1) * m;  // Offset for the i^th row

        for (j = p; j > 0; j--) {
            sum = 0;

            for (k = m; k > 0; k--) {
                innerRow = (k - 1) * p;  // Offset for the k^th row
                prod = fp16_mul(mat1->data[outerRow + (k - 1)], mat2->data[innerRow + (j - 1)], precision);
                sum = fp16_add(sum, prod);
            }
 
            resultRow = (i - 1) * p;
            result->data[resultRow + (j - 1)] = sum;
        }
    }
    #endif

    return result;
}


int16_t dot_product(Matrix *vec1, Matrix *vec2, uint16_t precision) {
    /**
     * Computes the dot product for the two vectors. If the inputs are not
     * proper vectors, then we use the first row of vec1 and first column of vec2.
     */
    uint16_t i, j;
    uint16_t vec1Idx, vec2Idx;
    int16_t result = 0;

    for (i = vec1->numCols; i > 0; i--) {
        vec1Idx = i - 1;
        vec2Idx = vec2->numCols * (i - 1);

        result = fp16_add(result, fp16_mul(vec1->data[vec1Idx], vec2->data[vec2Idx], precision));
    }

    return result;
}


Matrix *shuffled_vector_hadamard(Matrix *result, Matrix *inputs, Matrix *weights, uint16_t *indices, uint16_t precision) {
    /**
     * Computes the element-wise product of the weights and inputs where the input
     * elements are shuffled according to the given indices.
     */
    if ((result->numRows != inputs->numRows) || (inputs->numRows != weights->numRows)) {
        return NULL_PTR;
    }

    uint16_t i, j, k;
    for (i = result->numRows; i > 0; i--) {
        j = i - 1;
        k = VECTOR_INDEX(j);
        result->data[k] = fp16_mul(inputs->data[VECTOR_INDEX(indices[j])], weights->data[j], precision);
    }

    return result;
}


Matrix *apply_elementwise(Matrix *result, Matrix *mat, int16_t (*fn)(int16_t, uint16_t), uint16_t precision) {
    /**
     * Applies the given function to every element of the
     * input Matrix. Result stored directly in the matrix.
     */
    // Validate dimensions
    if ((result->numRows != mat->numRows) || (result->numCols != mat->numCols)) {
        return NULL_PTR;
    }

    uint16_t i;
    for (i = mat->numRows * mat->numCols; i > 0; i--) {
        result->data[i - 1] = (*fn)(mat->data[i - 1], precision);
    }

    return result;
}


Matrix *matrix_replace(Matrix *dst, Matrix *src) {
    /**
     * Replaces the contents of the destination Matrix with those from the src.
     */
    if ((dst->numRows != src->numRows) || (dst->numCols != src->numCols)) {
        return NULL_PTR;
    }

    uint16_t i, j;
    for (i = dst->numRows * dst->numCols; i > 0; i--) {
        j = i - 1;
        dst->data[j] = src->data[j];
    }

    return dst;
}


Matrix *matrix_set(Matrix *mat, int16_t value) {
    /**
     * Sets all values in the Matrix to the given value (already in fixed point form).
     */

    uint16_t i;
    for (i = mat->numRows * mat->numCols; i > 0; i--) {
        mat->data[i - 1] = value;
    }

    return mat;
}


int16_t argmax(Matrix *vec) {
    /**
     * Computes the argmax of the 1d vector. If the input has multiple columns, then
     * this is the argmax over the 1st column.
     */
    if (vec->numRows <= 0) {
        return -1;
    }

    uint16_t numCols = vec->numCols;

    int16_t max = vec->data[0];
    int16_t max_index = 0;

    uint16_t i;
    int16_t val;
    for (i = vec->numRows - 1; i > 0; i--) {
        val = vec->data[i * numCols];
        if (val > max) {
            max_index = i;
            max = val;
        }
    }

    return max_index;
}


/**
 * SPARSE MATRIX OPERATIONS
 */
Matrix *sp_matrix_vector_prod(Matrix *result, SparseMatrix *sp, Matrix *vec, uint16_t precision) {
    /**
     * Multiplies the given COO sparse matrix with a dense vector.
     */
    if ((result->numCols != VECTOR_COLS) || (vec->numCols != VECTOR_COLS)) {
        return NULL_PTR;
    }

    // Zero out the result matrix
    result = matrix_set(result, 0);

    uint16_t row, col;
    uint16_t rowIdx, dataIdx;
    uint16_t start, end, diff, offset;
    int16_t mul, rowSum;

    dataIdx = sp->nnz - 1;

    for (rowIdx = sp->numRows; rowIdx > 0; rowIdx--) {
        row = rowIdx - 1;

        start = sp->rowPtr[row];
        end = sp->rowPtr[rowIdx];

        //if (rowIdx < sp->numRows) {
        //    end = sp->rowPtr[rowIdx];
        //} else {
        //    end = sp->nnz;
        //}

        diff = end - start;
        rowSum = 0;

        for (offset = diff; offset > 0; offset--) {
            col = VECTOR_INDEX(sp->cols[start + offset - 1]);
    
            mul = fp16_mul(sp->data[dataIdx], vec->data[col], precision);
            rowSum = fp16_add(rowSum, mul);
            dataIdx -= 1;
        }

        result->data[VECTOR_INDEX(row)] = rowSum;
    }

    return result;
}


/**
 * BLOCK SPARSE MATRIX OPERATIONS
 */
Matrix *block_sparse_matrix_vector_prod(Matrix *result, BlockSparseMatrix *bsm, Matrix *vec, uint16_t precision) {
    // Validate arguments
    if ((bsm->numCols != vec->numRows) || (vec->numCols != VECTOR_COLS) || (bsm->numRows != result->numRows) || (result->numCols != VECTOR_COLS)) {
        return NULL_PTR;
    }

    // Zero out the result matrix
    result = matrix_set(result, 0);

    #ifdef IS_MSP
    // Create temporary buffers for the LEA values
    volatile uint16_t bufferOffset = 4;
    int16_t *tempOutput = MULTIPLY_BUFFER + bufferOffset;
    bufferOffset += bsm->numRows * VECTOR_COLS;

    int16_t *tempInput = MULTIPLY_BUFFER + bufferOffset;
    bufferOffset += bsm->numCols * VECTOR_COLS;

    int16_t *blockData = MULTIPLY_BUFFER + bufferOffset;

    uint16_t i, j, k;
    uint16_t numElements, inputOffset, outputOffset;
    Matrix *block;

    for (i = bsm->numBlocks; i > 0; i--) {
        // Load the block into LEA RAM via DMA
        j = i - 1;
        block = bsm->blocks[j];

        numElements = block->numRows * block->numCols;
        dma_load(blockData, block->data, numElements);

        inputOffset = bsm->cols[j];
        outputOffset = bsm->rows[j];

        // Load the input vector into LEA RAM via DMA
        numElements = block->numCols * vec->numCols;
        dma_load(tempInput, vec->data + VECTOR_INDEX(inputOffset), numElements);

        // Use the LEA to perform the matrix-vector product
        msp_status status;
        msp_matrix_mpy_q15_params mulParams;

        // Initialze LEA metadata
        mulParams.srcARows = block->numRows;
        mulParams.srcACols = block->numCols;
        mulParams.srcBRows = block->numCols;
        mulParams.srcBCols = VECTOR_COLS;

        // Perform Matrix multiplication using the LEA
        status = msp_matrix_mpy_q15(&mulParams, blockData, tempInput, tempOutput);
        msp_checkStatus(status);

        // Convert back to the original fixed-point precision. The LEA assumes 15 fractional bits.
        msp_matrix_shift_q15_params shiftParams;
        shiftParams.rows = block->numRows;
        shiftParams.cols = VECTOR_COLS;
        shiftParams.shift = 15 - precision;

        // Perform element-wise shift using the LEA
        if (shiftParams.shift > 0) {
            status = msp_matrix_shift_q15(&shiftParams, tempOutput, tempOutput);
            msp_checkStatus(status);
        }

        // Add elements to the result array
        for (k = block->numRows; k > 0; k--) {
            j = VECTOR_INDEX(k - 1 + outputOffset);
            result->data[j] = fp16_add(tempOutput[VECTOR_INDEX(k-1)], result->data[j]);
        }
    }

    #else
    
    uint16_t i, j, r, c;
    uint16_t n, m;
    uint16_t inputOffset, outputOffset;
    Matrix *block;
   
    int16_t sum, prod;
    uint16_t innerRow, outerRow, resultRow;

    for (i = bsm->numBlocks; i > 0; i--) {

        // Get the current block 
        j = i - 1;
        block = bsm->blocks[j];

        n = block->numRows;
        m = block->numCols;

        inputOffset = bsm->cols[j];
        outputOffset = bsm->rows[j];

        for (r = n; r > 0; r--) {
            outerRow = (r - 1) * m;  // Offset for the i^th row

            sum = 0;
            prod = 1;

            for (c = m; c > 0; c--) {
                innerRow = VECTOR_INDEX(c - 1 + inputOffset);  // Offset for the k^th row
                prod = fp16_mul(block->data[outerRow + (c - 1)], vec->data[innerRow], precision);
                sum = fp16_add(sum, prod);
            }
 
            resultRow = VECTOR_INDEX(r - 1 + outputOffset);
            result->data[resultRow] = fp16_add(sum, result->data[resultRow]);
        }
    }
       
    #endif

    return result;
}
