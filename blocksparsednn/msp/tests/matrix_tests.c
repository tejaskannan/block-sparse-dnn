#include "matrix_tests.h"


int main(void) {

    // Matrix addition
    printf("---- Testing Matrix Addition ----\n");
    test_add_two();
    test_add_three();
    test_add_diff();
    test_add_min_dims();
    test_add_wrong_dims();
    printf("\tPassed addition tests.\n");

    // Matrix Multiplication
    printf("---- Testing Matrix Multiplication ----\n");
    test_mult_two();
    test_mult_three();
    test_mult_diff();
    test_mult_vec();
    test_mult_wrong_dims();
    printf("\tPassed multiplication tests.\n");

    // Vector Dot Product
    printf("---- Testing Vector Dot Product ----\n");
    test_dot_product();
    test_dot_product_two();
    printf("\tPassed dot product tests.\n");

    // Matrix apply element-wise sigmoid
    printf("---- Testing Apply Element-Wise ----\n");
    test_apply_relu();
    printf("\tPassed apply element-wise tests.\n");

    // Matrix Replace
    printf("---- Testing Matrix Replace ----\n");
    test_replace();
    test_replace_wrong_dims();
    printf("\tPassed replacement tests.\n");

    // Vector Argmax
    printf("---- Testing Vector Argmax ----\n");
    test_argmax();
    test_argmax_two();
    printf("\tPassed argmax tests.\n");

    // Sparse Matrix Vector Product
    printf("---- Testing Sparse Matrix Vector Multiplication ----\n");
    test_sp_mult_three();
    test_sp_mult_three_frac();
    test_sp_mult_unequal();
    printf("\tPassed Sparse Mult tests.\n");

    // Block Sparse Matrix Vector Product
    printf("---- Testing Block Sparse Matrix Vector Multiplication ----\n");
    test_bs_mult_four();
    test_bs_mult_six();
    test_bs_mult_expand();
    test_bs_mult_contract();
    printf("\tPassed Block Sparse Mult tests.\n");

    // Shuffled Element-wise Products
    printf("---- Testing Shuffled Element-wise Multiplication ----\n");
    test_shuffled_mult_four();
    test_shuffled_mult_six();

    printf("--------------------\n");
    printf("Completed all tests.\n");
    return 0;
}


void test_add_two(void) {
    // Test 2 x 2 cases
    int16_t mat1Data[] = { 1, 2, 3, 4 };
    Matrix mat1 = { to_fixed_point(mat1Data, 4, PRECISION), 2, 2};
    
    int16_t mat2Data[] = { 5, 6, 7, 8 };
    Matrix mat2 = { to_fixed_point(mat2Data, 4, PRECISION), 2, 2 };

    int16_t expectedData[] = { 6, 8, 10, 12 };
    Matrix expected = { to_fixed_point(expectedData, 4, PRECISION), 2, 2 };

    Matrix *sum = matrix_add(&mat1, &mat1, &mat2);
    assert(matrix_equal(&expected, sum));
}


void test_add_three(void) {
    // Test 3 x 3 cases
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    Matrix mat1 = { to_fixed_point(mat1Data, 9, PRECISION), 3, 3 };
    
    int16_t mat2Data[] = { 11, 12, 13, 14, 15, 16, 17, 18, 19 };
    Matrix mat2 = { to_fixed_point(mat2Data, 9, PRECISION), 3, 3 };

    int16_t expectedData[] = { 12, 14, 16, 18, 20, 22, 24, 26, 28 };
    Matrix expected = { to_fixed_point(expectedData, 9, PRECISION), 3, 3 };

    Matrix *sum = matrix_add(&mat1, &mat1, &mat2);
    assert(matrix_equal(&expected, sum));
}

void test_add_diff(void) {
    // Test 2 x 3 * 3 x 3 case
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6 };
    Matrix mat1 = { to_fixed_point(mat1Data, 6, PRECISION), 2, 3 };

    int16_t mat2Data[] = { 7, 10, 11, 8, 12, 13 };
    Matrix mat2 = { to_fixed_point(mat2Data, 6, PRECISION), 2, 3 };

    int16_t expectedData[] = { 8, 12, 14, 12, 17, 19 };
    Matrix expected = { to_fixed_point(expectedData, 6, PRECISION), 2, 3 };

    Matrix *sum = matrix_add(&mat1, &mat1, &mat2);
    assert(matrix_equal(&expected, sum));
}


void test_add_min_dims(void) {
    // Test 4 x 1 + 4 x 2 case (only adds first column)
    int16_t mat1Data[] = { 1, 2, 3, 4 };
    Matrix mat1 = { to_fixed_point(mat1Data, 4, PRECISION), 4, 1 };

    int16_t mat2Data[] = { 7, 10, 11, 8, 12, 13, 15, 3 };
    Matrix mat2 = { to_fixed_point(mat2Data, 8, PRECISION), 4, 2 };

    int16_t expectedData[] = { 8, 13, 15, 19 };
    Matrix expected = { to_fixed_point(expectedData, 4, PRECISION), 4, 1 };

    Matrix *sum = matrix_add(&mat1, &mat1, &mat2);
    assert(matrix_equal(&expected, sum));
}


void test_add_wrong_dims(void) {
    // Test misaligned dimensions
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    Matrix mat1 = { to_fixed_point(mat1Data, 9, PRECISION), 3, 3 };

    int16_t mat2Data[] = { 4, 5, 6, 7 };
    Matrix mat2 = { to_fixed_point(mat2Data, 4, PRECISION), 2, 2 };

    assert(matrix_add(&mat1, &mat1, &mat2) == NULL_PTR);
    assert(matrix_add(&mat2, &mat2, &mat1) == NULL_PTR);
}


void test_mult_two(void) {
    // Test 2 x 2 cases
    int16_t mat1Data[] = { 1, 2, 3, 4 };
    Matrix mat1 = { to_fixed_point(mat1Data, 4, PRECISION), 2, 2 };

    int16_t mat2Data[] = { 5, 6, 7, 8 };
    Matrix mat2 = { to_fixed_point(mat2Data, 4, PRECISION), 2, 2 };

    int16_t expectedData[] = { 19, 22, 43, 50 };
    Matrix expected = { to_fixed_point(expectedData, 4, PRECISION), 2, 2 };

    int16_t resultData[4] = { 0 };
    Matrix result = { resultData, 2, 2 };

    matrix_multiply(&result, &mat1, &mat2, PRECISION);
    assert(matrix_equal(&expected, &result));
}

void test_mult_three(void) {
    // Test 3 x 3 cases
    int16_t precision = 5;

    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    Matrix mat1 = { to_fixed_point(mat1Data, 9, precision), 3, 3 };

    int16_t mat2Data[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
    Matrix mat2 = { to_fixed_point(mat2Data, 9, precision), 3, 3 };

    int16_t expectedData[] = { 84, 90, 96, 201, 216, 231, 318, 342, 366 };
    Matrix expected = { to_fixed_point(expectedData, 9, precision), 3, 3 };

    int16_t resultData[9] = { 0 };
    Matrix result = { resultData, 3, 3 };
    
    matrix_multiply(&result, &mat1, &mat2, precision);
    assert(matrix_equal(&expected, &result));
}


void test_mult_diff(void) {
    // Test 2 x 3 * 3 x 3 case
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6 };
    Matrix mat1 = { to_fixed_point(mat1Data, 6, PRECISION), 2, 3 };

    int16_t mat2Data[] = { 7, 10, 11, 8, 12, 13, 9, 14, 15 };
    Matrix mat2 = { to_fixed_point(mat2Data, 9, PRECISION), 3, 3 };

    int16_t expectedData[] = { 50, 76, 82, 122, 184, 199 };
    Matrix expected = { to_fixed_point(expectedData, 6, PRECISION), 2, 3 };

    int16_t resultData[6] = { 0 };
    Matrix result = { resultData, 2, 3 };

    matrix_multiply(&result, &mat1, &mat2, PRECISION);
    assert(matrix_equal(&expected, &result));
}


void test_mult_vec(void) {
    int16_t matData[] = { 2, 3, 4, 5, 1, 2 };
    Matrix mat = { to_fixed_point(matData, 6, PRECISION), 2, 3 };

    int16_t vecData[] = { 2, 3, 4 };
    Matrix vec = { to_fixed_point(vecData, 3, PRECISION), 3, 1 };

    int16_t expectedData[] = { 29, 21 };
    Matrix expected = { to_fixed_point(expectedData, 2, PRECISION), 2, 1 };

    int16_t resultData[2] = { 0 };
    Matrix result = { resultData, 2, 1 };

    matrix_multiply(&result, &mat, &vec, PRECISION);
    assert(matrix_equal(&expected, &result));
}


void test_mult_wrong_dims(void) {
    // Test cases where dimensions are not aligned
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6 };
    Matrix mat1 = { mat1Data, 2, 3 };

    int16_t mat2Data[] = { 7, 10, 11, 12 };
    Matrix mat2 = { mat2Data, 4, 1 };

    int16_t resultData[9] = { 0 };
    Matrix result = { resultData, 3, 3 };

    assert(matrix_multiply(&result, &mat1, &mat2, PRECISION) == NULL_PTR);
    assert(matrix_multiply(&result, &mat2, &mat1, PRECISION) == NULL_PTR);
}


void test_sp_mult_three(void) {
    int16_t vecData[] = {1, -2, 3};
    Matrix vec = { vecData, 3, 1 };

    int16_t spData[] = { -1, -1, 1, -3 };
    //uint16_t rows[] = { 1, 1, 2, 2 };
    uint16_t rowPtr[] = { 0, 0, 2, 4 };
    uint16_t cols[] = { 0, 1, 1, 2 };
    SparseMatrix sp = { spData, 3, 3, rowPtr, cols, 4 };

    int16_t resultData[3];
    Matrix result = { resultData, 3, 1 };

    sp_matrix_vector_prod(&result, &sp, &vec, 0);

    int16_t expectedData[] = { 0, 1, -11 };
    Matrix expected = { expectedData, 3, 1};

    assert(matrix_equal(&expected, &result));
}


void test_sp_mult_three_frac(void) {
    int16_t one = 1 << PRECISION;
    int16_t half = 1 << (PRECISION - 1);
    int16_t fourth = 1 << (PRECISION - 2);
    int16_t two = 1 << (PRECISION + 1);

    int16_t vecData[] = { one, -1 * half, fourth };
    Matrix vec = { vecData, 3, 1 };

    int16_t spData[] = { one, half, two, fourth };
    // uint16_t rows[] = { 0, 1, 1, 2 };
    uint16_t rowPtr[] = { 0, 1, 3, 4 };
    uint16_t cols[] = { 0, 1, 2, 0 };
    SparseMatrix sp = { spData, 3, 3, rowPtr, cols, 4 };

    int16_t resultData[3];
    Matrix result = { resultData, 3, 1 };

    sp_matrix_vector_prod(&result, &sp, &vec, 7);

    int16_t expectedData[] = { one, fourth, fourth };
    Matrix expected = { expectedData, 3, 1};

    assert(matrix_equal(&expected, &result));
}


void test_sp_mult_unequal(void) {
    int16_t one = 1 << PRECISION;
    int16_t two = 1 << (PRECISION + 1);
    int16_t three = 3 * one;
    int16_t half = 1 << (PRECISION - 1);

    int16_t vecData[] = { half, -1 * one, one };
    Matrix vec = { vecData, 3, 1 };

    int16_t spData[] = { -1 * three, -1 * two, two, -1 * two, two, -1 * three, two, -1 * three };
    // uint16_t rows[] = { 0, 0, 1, 1, 2, 3, 3, 3 };
    uint16_t rowPtr[] = { 0, 2, 4, 5, 8 };
    uint16_t cols[] = { 0, 1, 1, 2, 1, 0, 1, 2 };
    SparseMatrix sp = { spData, 4, 3, rowPtr, cols, 8 };

    int16_t resultData[4];
    Matrix result = { resultData, 4, 1 };

    sp_matrix_vector_prod(&result, &sp, &vec, 7);

    int16_t expectedData[] = { half, int_to_fp16(-4, PRECISION), -1 * two, float_to_fp16(-6.5, PRECISION) };
    Matrix expected = { expectedData, 4, 1};

    assert(matrix_equal(&expected, &result));
}


void test_bs_mult_four(void) {
    // Make the block sparse matrix
    int16_t matData[] = { 1, 2, 3, 4 };
    Matrix block = { matData, 2, 2 };
    Matrix *blocks[] = { &block, &block };

    uint16_t rows[] = { 0, 2 };
    uint16_t cols[] = { 0, 2 };

    BlockSparseMatrix bsm = { blocks, 2, 4, 4, rows, cols };

    int16_t vecData[] = { 1, 2, 3, 4 };
    Matrix vec = { vecData, 4, 1 };

    int16_t resultData[4];
    Matrix result = { resultData, 4, 1 };

    block_sparse_matrix_vector_prod(&result, &bsm, &vec, 0);

    int16_t expectedData[] = { 5, 11, 11, 25 };
    Matrix expected = { expectedData, 4, 1 };

    assert(matrix_equal(&expected, &result));
}


void test_bs_mult_six(void) {
    // Make constants
    uint16_t precision = 10;
    int16_t one = 1 << precision;
    int16_t two = 1 << (precision + 1);
    int16_t half = 1 << (precision - 1);
    int16_t fourth = 1 << (precision - 2);
    int16_t three_fourths = half + fourth;

    // Make the block sparse matrix
    int16_t block1Data[] = { one, two, one, half, fourth, -1 * fourth };
    Matrix block1 = { block1Data, 2, 3 };

    int16_t block2Data[] = { -1 * half, one, -1 * fourth, half + fourth, one, -1 * (one + half) };
    Matrix block2 = { block2Data, 2, 3 };

    Matrix *blocks[] = { &block1, &block2 };

    uint16_t rows[] = { 1, 2 };
    uint16_t cols[] = { 0, 3 };

    BlockSparseMatrix bsm = { blocks, 2, 6, 6, rows, cols };

    int16_t vecData[] = { half, fourth, -1 * one, one + half, -1 * two, 0 };
    Matrix vec = { vecData, 6, 1 };

    int16_t resultData[6];
    Matrix result = { resultData, 6, 1 };

    block_sparse_matrix_vector_prod(&result, &bsm, &vec, precision);

    int16_t expectedData[] = { 0, 0, -2240, -896, 0, 0 };
    Matrix expected = { expectedData, 6, 1 };

    assert(matrix_equal(&expected, &result));
}


void test_bs_mult_expand(void) {
    // Make constants
    uint16_t precision = 10;
    int16_t one = 1 << precision;
    int16_t two = 1 << (precision + 1);
    int16_t half = 1 << (precision - 1);
    int16_t fourth = 1 << (precision - 2);
    int16_t three_fourths = half + fourth;

    // Make the block sparse matrix
    int16_t block1Data[] = { one, two, one, half, fourth, -1 * fourth };
    Matrix block1 = { block1Data, 2, 3 };

    int16_t block2Data[] = { -1 * half, one, half + fourth, one };
    Matrix block2 = { block2Data, 2, 2 };

    Matrix *blocks[] = { &block1, &block2 };

    uint16_t rows[] = { 1, 2 };
    uint16_t cols[] = { 0, 3 };

    BlockSparseMatrix bsm = { blocks, 2, 6, 5, rows, cols };

    int16_t vecData[] = { half, fourth, -1 * one, one + half, -1 * two };
    Matrix vec = { vecData, 5, 1 };

    int16_t resultData[6];
    Matrix result = { resultData, 6, 1 };

    block_sparse_matrix_vector_prod(&result, &bsm, &vec, precision);

    int16_t expectedData[] = { 0, 0, -2240, -896, 0, 0 };
    Matrix expected = { expectedData, 6, 1 };

    assert(matrix_equal(&expected, &result));
}


void test_bs_mult_contract(void) {
    // Make constants
    uint16_t precision = 10;
    int16_t one = 1 << precision;
    int16_t two = 1 << (precision + 1);
    int16_t half = 1 << (precision - 1);
    int16_t fourth = 1 << (precision - 2);
    int16_t three_fourths = half + fourth;

    // Make the block sparse matrix
    int16_t block1Data[] = { one, two, one, half, fourth, -1 * fourth };
    Matrix block1 = { block1Data, 2, 3 };

    int16_t block2Data[] = { -1 * half, one, -1 * fourth, half + fourth, one, -1 * (one + half) };
    Matrix block2 = { block2Data, 2, 3 };

    Matrix *blocks[] = { &block1, &block2 };

    uint16_t rows[] = { 1, 2 };
    uint16_t cols[] = { 0, 3 };

    BlockSparseMatrix bsm = { blocks, 2, 5, 6, rows, cols };

    int16_t vecData[] = { half, fourth, -1 * one, one + half, -1 * two, 0 };
    Matrix vec = { vecData, 6, 1 };

    int16_t resultData[6];
    Matrix result = { resultData, 5, 1 };

    block_sparse_matrix_vector_prod(&result, &bsm, &vec, precision);

    int16_t expectedData[] = { 0, 0, -2240, -896, 0 };
    Matrix expected = { expectedData, 5, 1 };

    assert(matrix_equal(&expected, &result));
}

void test_apply_relu(void) {
    int16_t matData[6] = { 0, 1, -2, 2, -1, 5 };
    Matrix mat = { matData, 2, 3 };

    int16_t expectedData[6] = { 0, 1, 0, 2, 0, 5 };
    Matrix expected = { expectedData, 2, 3 };

    Matrix *result = apply_elementwise(&mat, &mat, &fp16_relu, PRECISION);
    assert(matrix_equal(&expected, result));
}


void test_replace(void) {
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6 };
    Matrix mat1 = { to_fixed_point(mat1Data, 6, PRECISION), 2, 3 };

    int16_t mat2Data[6] = { 0 };
    Matrix mat2 = { mat2Data, 2, 3 };
    
    matrix_replace(&mat2, &mat1);
    assert(matrix_equal(&mat1, &mat2));
}


void test_replace_wrong_dims(void) {
    // Test misaligned dimensions
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    Matrix mat1 = { to_fixed_point(mat1Data, 9, PRECISION), 3, 3 };

    int16_t mat2Data[6] = { 0 };
    Matrix mat2 = { mat2Data, 2, 3 };
    assert(matrix_replace(&mat2, &mat1) == NULL_PTR);
}


void test_argmax(void) {
    int16_t vecData[] = { 2, 3, 1, 4, 5, 2 };
    Matrix vec = { to_fixed_point(vecData, 6, PRECISION), 6, 1 };
    assert(4 == argmax(&vec));
}


void test_argmax_two(void) {
    int16_t vecData[] = { 2, 3, 1, 4, 5, 6 };
    Matrix vec = { to_fixed_point(vecData, 6, PRECISION), 3, 2 };
    assert(2 == argmax(&vec));
}


void test_dot_product(void) {
    int16_t vec1Data[] = { 2, 3, 1 };
    Matrix vec1 = { to_fixed_point(vec1Data, 3, PRECISION), 1, 3 };

    int16_t vec2Data[] = { 4, 5, 2 };
    Matrix vec2 = { to_fixed_point(vec2Data, 3, PRECISION), 3, 1 };

    int16_t result = dot_product(&vec1, &vec2, PRECISION);
    assert(int_to_fp16(25, PRECISION) == result);
}


void test_dot_product_two(void) {
    int16_t vec1Data[] = { 2, 3, 1, 1, 1, 1 };
    Matrix vec1 = { to_fixed_point(vec1Data, 3, PRECISION), 1, 3 };

    int16_t vec2Data[] = { 4, 1, 5, 1, 2, 1 };
    Matrix vec2 = { to_fixed_point(vec2Data, 6, PRECISION), 3, 2 };

    int16_t result = dot_product(&vec1, &vec2, PRECISION);
    assert(int_to_fp16(25, PRECISION) == result);
}


void test_shuffled_mult_four(void) {
    uint16_t indices[] = { 1, 3, 2, 0 };
    
    int16_t one = (1 << PRECISION);
    int16_t two = (1 << (PRECISION + 1));
    int16_t half = (1 << (PRECISION - 1));
    int16_t fourth = (1 << (PRECISION - 2));

    int16_t inputData[] = { -1 * one, one, two, half };
    Matrix inputs = { inputData, 4, 1 };

    int16_t weightData[] = { two, -1 * half, fourth, one };
    Matrix weights = { weightData, 4, 1 };

    int16_t resultData[4];
    Matrix result = { resultData, 4, 1 };

    shuffled_vector_hadamard(&result, &inputs, &weights, indices, PRECISION);

    int16_t expectedData[] = { two, -1 * fourth, half, -1 * one};
    Matrix expected = { expectedData, 4, 1 };

    assert(matrix_equal(&result, &expected));
}

void test_shuffled_mult_six(void) {
    uint16_t indices[] = { 1, 4, 0, 5, 2, 3 };
    
    int16_t one = (1 << PRECISION);
    int16_t two = (1 << (PRECISION + 1));
    int16_t four = (1 << (PRECISION + 2));
    int16_t half = (1 << (PRECISION - 1));
    int16_t fourth = (1 << (PRECISION - 2));
    int16_t eighth = (1 << (PRECISION - 3));

    int16_t inputData[] = { -1 * one, -1 * two, -1 * half, half, four, one };
    Matrix inputs = { inputData, 6, 1 };

    int16_t weightData[] = { two, -1 * half, fourth, one, two, -1 * fourth};
    Matrix weights = { weightData, 6, 1 };

    int16_t resultData[6];
    Matrix result = { resultData, 6, 1 };

    shuffled_vector_hadamard(&result, &inputs, &weights, indices, PRECISION);

    int16_t expectedData[] = { -1 * four, -1 * two, -1 * fourth, one, -1 * one, -1 * eighth};
    Matrix expected = { expectedData, 6, 1 };

    assert(matrix_equal(&result, &expected));
}



int matrix_equal(Matrix *mat1, Matrix *mat2) {
    if (mat1 == NULL_PTR && mat2 == NULL_PTR) {
        return 1;
    }

    if (mat1 == NULL_PTR && mat2 != NULL_PTR) {
        return 0;
    }

    if (mat1 != NULL_PTR && mat2 == NULL_PTR) {
        return 0;
    }
    
    if ((mat1->numRows != mat2->numRows) || (mat1->numCols != mat2->numCols)) {
        return 0;
    }

    for (int i = 0; i < mat1->numRows * mat2->numCols; i++) {
        if (mat1->data[i] != mat2->data[i]) {
            return 0;
        }
    }

    return 1;
}


int16_t *to_fixed_point(int16_t *data, uint16_t n, uint16_t precision) {
    for (int16_t i = 0; i < n; i++) {
        data[i] = int_to_fp16(data[i], precision);
    }

    return data;
}
