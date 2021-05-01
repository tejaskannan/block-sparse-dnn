#include <stdio.h>
#include <assert.h>
#include <stdint.h>

#include "../math/matrix.h"
#include "../math/matrix_ops.h"


#ifndef MATRIX_TEST_GUARD
    #define MATRIX_TEST_GUARD

    #define PRECISION 7

    // Utility Functions
    int matrix_equal(Matrix *mat1, Matrix *mat2);
    int16_t *to_fixed_point(int16_t *data, uint16_t n, uint16_t precision);

    void test_add_two(void);
    void test_add_three(void);
    void test_add_diff(void);
    void test_add_min_dims(void);
    void test_add_wrong_dims(void);
    void test_mult_two(void);
    void test_mult_three(void);
    void test_mult_diff(void);
    void test_mult_vec(void);
    void test_mult_wrong_dims(void);
    void test_dot_product(void);
    void test_dot_product_two(void);
    void test_apply_relu(void);
    void test_replace(void);
    void test_replace_wrong_dims(void);
    void test_argmax(void);
    void test_argmax_two(void);

    void test_sp_mult_three(void);
    void test_sp_mult_three_frac(void);
    void test_sp_mult_unequal(void);

#endif
