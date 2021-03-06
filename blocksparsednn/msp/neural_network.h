#include <stdint.h>
#include "math/matrix.h"
#include "neural_network_parameters.h"
#include "layers/layers.h"

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

    int16_t block_sparse_mlp(Matrix *inputs, uint16_t precision);
    int16_t sparse_mlp(Matrix *inputs, uint16_t precision);
    int16_t dense_mlp(Matrix *inputs, uint16_t precision);
    int16_t block_diagonal_mlp(Matrix *inputs, uint16_t precision);

#endif
