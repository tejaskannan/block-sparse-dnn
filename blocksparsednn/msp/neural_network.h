#include <stdint.h>
#include "math/matrix.h"
#include "neural_network_parameters.h"
#include "layers/layers.h"

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

    Matrix *block_sparse_mlp(Matrix *result, Matrix *inputs, uint16_t precision);

#endif
