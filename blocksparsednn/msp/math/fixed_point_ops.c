#include "fixed_point_ops.h"


int16_t fp16_add(int16_t x, int16_t y) {
    return x + y;
}


int16_t fp16_sub(int16_t x, int16_t y) {
    return x - y;
}


int16_t fp16_mul(int16_t x, int16_t y, uint16_t precision) {
    int32_t mul = ((int32_t) x) * ((int32_t) y);
    return (int16_t) (mul >> precision);
}


int16_t fp16_div(int16_t x, int16_t y, uint16_t precision) {
    int32_t xLarge = ((int32_t) x) << precision;
    return (int16_t) (xLarge / y);
}


int16_t fp16_neg(int16_t x) {
    return -1 * x;
}


int16_t float_to_fp16(float x, uint16_t precision) {
    return (int16_t) (x * (1 << precision));
}


int16_t int_to_fp16(int16_t x, uint16_t precision) {
    return x * (1 << precision);
}


int16_t fp16_relu(int16_t x, uint16_t precision) {
    UNUSED(precision);
    if (x >= 0) {
        return x;
    }
    return 0;
}


int16_t fp16_leaky_relu(int16_t x, uint16_t precision) {
    UNUSED(precision);
    int16_t isPositive = (int16_t) (x > 0);

    // We perform division by 4 like this because bit shifting
    // is more efficient than division
    int16_t leakyX = (x >> 2);
    return isPositive * x + (1 - isPositive) * leakyX;
}


int16_t fp16_linear(int16_t x, uint16_t precision) {
    UNUSED(precision);
    return x;
}


// 32 bit fixed-point operations
int32_t fp32_add(int32_t x, int32_t y) {
    return x + y;
}


int32_t fp32_neg(int32_t x) {
    return -1 * x;
}


int32_t fp32_sub(int32_t x, int32_t y) {
    return fp32_add(x, fp32_neg(y));
}


int32_t fp32_mul(int32_t x, int32_t y, uint16_t precision) {
    int64_t xLarge = (int64_t) x;
    int64_t yLarge = (int64_t) y;

    return (int32_t) ((xLarge * yLarge) >> precision);
}


int32_t fp32_div(int32_t x, int32_t y, uint16_t precision) {
    int64_t xLarge = ((int64_t) x) << precision;
    return (int32_t) (xLarge / y);
}


int32_t int_to_fp32(int32_t x, uint16_t precision) {
    return ((int32_t) x) << precision;
}
