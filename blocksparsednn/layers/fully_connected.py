import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, List, Union

from utils.tf_utils import get_activation, project_block_mask, block_diagonal_matmul, block_sparse_matmul, tile_to_size
from utils.tf_utils import create_diagonal_pattern

try:
    from blocksparse.matmul import BlocksparseMatMul
except ImportError:
    pass


@tf.custom_gradient
def binary_round(inputs: tf.Tensor, threshold: float) -> tf.Tensor:
    """
    Rounds the values in a [0, 1] tensor to {0, 1} and defines
    the gradient using an estimator based on the following:
    http://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
    """
    def grad(dy):
        abs_inputs = tf.abs(inputs)
        cond = tf.cast(abs_inputs < 0.4, dtype=dy.dtype)
        grad_weight = (2.0 - 4.0 * abs_inputs) * cond + 0.4 * (1.0 - cond)
        return grad_weight * dy, tf.constant(0)
        # return dy, tf.constant(0)

    return tf.cast(tf.less(inputs, threshold), dtype=inputs.dtype), grad


def fully_connected(inputs: tf.Tensor,
                    units: int,
                    activation: Optional[str],
                    dropout_keep_rate: tf.Tensor,
                    use_bias: bool,
                    use_dropout: bool,
                    name: str) -> tf.Tensor:
    """
    Builds a fully connected (dense) neural network layer.

    Args:
        inputs: A [B, N] tensor containing the input features (N)
            for each batch element (B)
        units: The number of output units (M)
        activation: The name of the activation function. If none,
            no activation is applied
        dropout_keep_rate: A scalar containing the dropout keep rate
        use_bias: Whether to apply a bias term
        use_dropout: Whether to apply dropout
        name: The name of this layer
    """
    with tf.variable_scope(name):
        # Make the trainable weight matrix
        input_units = inputs.get_shape()[-1]
        W = tf.get_variable(name='kernel',
                                      shape=(input_units, units),  # [N, M]
                                      initializer=tf.glorot_uniform_initializer(),
                                      trainable=True)

        transformed = tf.matmul(inputs, W)  # [B, M]

        # Apply bias if required
        if use_bias:
            bias = tf.get_variable(name='bias',
                                             shape=(1, units),
                                             initializer=tf.random_uniform_initializer(minval=-0.7, maxval=0.7),
                                             trainable=True)
            transformed = transformed + bias

        # Apply the activation function
        activation_fn = get_activation(activation)

        if activation_fn is not None:
            output = activation_fn(transformed)
        else:
            output = transformed

        # Apply dropout if required
        if use_dropout:
            output = tf.nn.dropout(output, rate=1.0 - dropout_keep_rate)

        return output


def sparse_connected(inputs: tf.Tensor,
                     units: int,
                     activation: Optional[str],
                     dropout_keep_rate: tf.Tensor,
                     use_bias: bool,
                     use_dropout: bool,
                     weight_indices: Union[tf.Tensor, np.ndarray],
                     name: str):
    """
    Creates a fully connected layer with sparse connections.

    Args:
        inputs: A [B, N] tensor of input features (N) for each batch sample (B)
        units: The number of output units (M)
        activation: The name of the activation function. None implies linear activation
        dropout_keep_rate: The keep probability for dropout
        use_bias: Whether to apply a bias term
        use_dropout: Whether to apply a dropout term
        weight_indices: A 2d tensor of (row, col) indices of the weight terms
        name: The name prefix of this layer
    Returns:
        The transformed tensor, [B, M]
    """
    with tf.variable_scope(name):
        # Create the (sparse) weight matrix with dimensions [M, N]
        num_nonzero = weight_indices.get_shape()[0] if isinstance(weight_indices, tf.Tensor) else weight_indices.shape[0]  # K
        weights = tf.get_variable(name='kernel',
                                            shape=(num_nonzero, ),
                                            initializer=tf.glorot_uniform_initializer(),
                                            trainable=True)

        input_units = inputs.get_shape()[-1]  # N
        dense_shape = (units, input_units)

        # Perform the sparse matrix multiplication via an embedding lookup
        #inputs_T = tf.transpose(inputs, perm=[1, 0])  # [N, B]

        #sp_ids = tf.sparse.SparseTensor(indices=weight_indices,
        #                                values=weight_indices[:, 1],
        #                                dense_shape=dense_shape) # Output IDs are the columns

        #sp_weights = tf.sparse.SparseTensor(indices=weight_indices,
        #                                    values=weights,
        #                                    dense_shape=dense_shape)

        ## [M, B]
        #transformed_T = tf.nn.embedding_lookup_sparse(params=inputs_T,
        #                                              sp_ids=sp_ids,
        #                                              sp_weights=sp_weights,
        #                                              combiner='sum')

        #transformed = tf.transpose(transformed_T, perm=[1, 0])

        # Create the sparse weight matrix
        weight_mat = tf.SparseTensor(indices=weight_indices,
                                     values=weights,
                                     dense_shape=dense_shape)  # [M, N]

        # Apply the weight matrix via a sparse matrix multiplication, [M, B]
        transp_transformed = tf.sparse.sparse_dense_matmul(weight_mat, inputs, adjoint_b=True)  # [M, B]
        transformed = tf.transpose(transp_transformed, perm=[1, 0])  # [B, M]

        # Apply bias if required
        if use_bias:
            bias = tf.get_variable(name='bias',
                                             shape=(1, units),
                                             initializer=tf.random_uniform_initializer(minval=-0.7, maxval=0.7),
                                             trainable=True)
            transformed = transformed + bias

        # Apply the activation function
        activation_fn = get_activation(activation)

        if activation_fn is not None:
            output = activation_fn(transformed)
        else:
            output = transformed

        # Apply dropout if required
        if use_dropout:
            output = tf.nn.dropout(output, rate=1.0 - dropout_keep_rate)

        return output


def block_masked_fully_connected(inputs: tf.Tensor,
                                 units: int,
                                 activation: Optional[str],
                                 dropout_keep_rate: tf.Tensor,
                                 use_bias: bool,
                                 use_dropout: bool,
                                 block_size: int,
                                 sparsity: float,
                                 name: str):
    """
    Creates a fully connected layer with sparse connections.

    Args:
        inputs: A [B, N] tensor of input features (N) for each batch sample (B)
        units: The number of output units (M)
        activation: The name of the activation function. None implies linear activation
        dropout_keep_rate: The keep probability for dropout
        use_bias: Whether to apply a bias term
        use_dropout: Whether to apply a dropout term
        block_size: The block size. Must be a divisor of N and M.
        sparsity: The nonzero fraction
        name: The name prefix of this layer
    Returns:
        The transformed tensor, [B, M]
    """
    with tf.variable_scope(name):
        # Make the trainable weight matrix
        input_units = inputs.get_shape()[-1]
        W = tf.get_variable(name='kernel',
                                      shape=(input_units, units),  # [N, M]
                                      initializer=tf.glorot_uniform_initializer(),
                                      dtype=inputs.dtype,
                                      trainable=True)

        # Make the trainable block mask
        block_pattern = tf.get_variable(name='block-mask',
                                                  shape=(input_units // block_size, units // block_size),
                                                  initializer=tf.glorot_uniform_initializer(),
                                                  dtype=inputs.dtype,
                                                  trainable=True)  # [N / K, M / K]

        # Turn the block pattern into a binary block mask
        block_mask = binary_round(tf.math.sigmoid(block_pattern), sparsity)  # [N / K, M / K]

        # Project the block mask up to the size of the weight variable, [N, M]
        block_mask = project_block_mask(block_mask, block_size=block_size)

        masked_weights = tf.multiply(W, block_mask)
        transformed = tf.matmul(inputs, masked_weights)  # [B, M]

        # Apply the diagonal transformation
        diag_weight = tf.compat.v1.get_variable('diagonal',
                                                shape=[1, 1],
                                                dtype=inputs.dtype,
                                                initializer=tf.compat.v1.glorot_uniform_initializer(),
                                                trainable=True)
        
        tiled_inputs = tile_to_size(inputs, size=units)  # [B, M]
        transformed_tiled = tf.multiply(tiled_inputs, diag_weight)  # [B, M]

        transformed = tf.add(transformed, transformed_tiled)

        # Apply bias if required
        if use_bias:
            bias = tf.get_variable(name='bias',
                                             shape=(1, units),
                                             initializer=tf.random_uniform_initializer(minval=-0.7, maxval=0.7),
                                             trainable=True)
            transformed = transformed + bias

        # Apply the activation function
        activation_fn = get_activation(activation)

        if activation_fn is not None:
            output = activation_fn(transformed)
        else:
            output = transformed

        # Apply dropout if required
        if use_dropout:
            output = tf.nn.dropout(output, rate=1.0 - dropout_keep_rate)

        return output, block_mask


def block_sparse_connected(inputs: tf.Tensor,
                           units: int,
                           activation: Optional[str],
                           dropout_keep_rate: tf.Tensor,
                           use_bias: bool,
                           use_dropout: bool,
                           nonzero_rows: Union[tf.Tensor, List[int]],
                           nonzero_cols: Union[tf.Tensor, List[int]],
                           block_size: int,
                           sparse_indices: List[int],
                           name: str):
    """
    Creates a fully connected layer with sparse connections.

    Args:
        inputs: A [B, N] tensor of input features (N) for each batch sample (B)
        units: The number of output units (M)
        activation: The name of the activation function. None implies linear activation
        dropout_keep_rate: The keep probability for dropout
        use_bias: Whether to apply a bias term
        use_dropout: Whether to apply a dropout term
        nonzero_rows: A [L] tensor holding the nonzero row indices
        nonzero_cols: A [L] tensor holding the nonzero column indices
        block_size: The block size (D)
        sparse_indices: A list of indices for the single sparse connections
        name: The name prefix of this layer
    Returns:
        The transformed tensor, [B, M]
    """
    in_units = inputs.get_shape()[-1]

    assert in_units > 0, 'Must provide a positive number of units.'
    assert units > 0, 'Must provide a positive number of output units.'

    if isinstance(nonzero_rows, list) and isinstance(nonzero_cols, list):
        assert len(nonzero_rows) == len(nonzero_cols), 'Must provide the same number of rows and columns'
        num_blocks = len(nonzero_rows)
    else:
        assert (nonzero_rows.get_shape() == nonzero_cols.get_shape()), 'Must provide same number of rows and columns.'
        num_blocks = nonzero_rows.get_shape()[0]

    with tf.variable_scope(name):
        
        weights: List[tf.Variable] = []
        for idx in range(num_blocks):
            weight = tf.get_variable('kernel-{0}'.format(idx),
                                               shape=[block_size, block_size],
                                               dtype=inputs.dtype,
                                               initializer=tf.glorot_uniform_initializer(),
                                               trainable=True)
            weights.append(weight)

        # Transform the input, [B, M]
        transformed = block_sparse_matmul(dense_mat=inputs,
                                          blocks=weights,
                                          nonzero_rows=nonzero_rows,
                                          nonzero_cols=nonzero_cols,
                                          output_dims=units)

        # Create the diagonal weight element
        random_conn = tf.get_variable('random-conn',
                                       shape=[1, units],
                                       dtype=inputs.dtype,
                                       initializer=tf.compat.v1.glorot_uniform_initializer(),
                                       trainable=True)

        # tiled_inputs = tile_to_size(inputs, size=units)  # [B, M]
        # transformed_tiled = tf.multiply(tiled_inputs, diag_weight)  # [B, M]
        gathered = tf.gather(inputs, indices=sparse_indices, axis=-1)  # [B, M]
        transformed_rand = tf.multiply(gathered, random_conn)

        transformed = tf.add(transformed, transformed_rand)

        # Apply bias if required
        if use_bias:
            bias = tf.get_variable(name='bias',
                                             shape=(1, units),
                                             initializer=tf.random_uniform_initializer(minval=-0.7, maxval=0.7),
                                             trainable=True)

            transformed = tf.add(transformed, bias)

        # Apply the activation function
        activation_fn = get_activation(activation)

        if activation_fn is not None:
            output = activation_fn(transformed)
        else:
            output = transformed

        # Apply dropout if required
        if use_dropout:
            output = tf.nn.dropout(output, rate=1.0 - dropout_keep_rate)

        return output


def block_diagonal_connected(inputs: tf.Tensor,
                             units: int,
                             in_units: int,
                             activation: Optional[str],
                             dropout_keep_rate: tf.Tensor,
                             use_bias: bool,
                             use_dropout: bool,
                             block_size: int,
                             sparse_indices: List[int],
                             name: str,
                             use_bsmm: bool):
    """
    Creates a fully connected layer with sparse connections.

    Args:
        inputs: A [B, N] tensor of input features (N) for each batch sample (B)
        units: The number of output units (M)
        in_units: The number of input units (N)
        activation: The name of the activation function. None implies linear activation
        dropout_keep_rate: The keep probability for dropout
        use_bias: Whether to apply a bias term
        use_dropout: Whether to apply a dropout term
        sparse_indices: A list of [M] indices denoting where to place the single random connections
        name: The name prefix of this layer
        use_bsmm: Whether to use the OpenAI block sparse matmul layer
    Returns:
        The transformed tensor, [B, M]
    """
    assert units > 0, 'Must provide a positive number of output units.'
    assert in_units > 0, 'Must provide a positive number of output units.'

    assert (units % block_size) == 0, 'Block Size ({0}) must divide the output units ({1}).'.format(block_size, units)
    assert (in_units % block_size) == 0, 'Block Size ({0}) must divide the input units ({1}).'.format(block_size, in_units)

    input_block_dim = int(in_units / block_size)
    output_block_dim = int(units / block_size)

    with tf.variable_scope(name):

        # Create the block diagonal pattern, accounting for
        # differences in the input and output dimension
        pattern = create_diagonal_pattern(input_dim=input_block_dim,
                                          output_dim=output_block_dim)

        if use_bsmm:
            bsmm = BlocksparseMatMul(pattern, block_size=block_size)
            weights = tf.get_variable('kernel',
                                      shape=bsmm.w_shape,
                                      dtype=tf.float32)

            print('Inputs: {0}'.format(inputs))

            transformed = bsmm(inputs, weights)  # [B, M]

            print('Transformed: {0}'.format(transformed))
        else:
            num_blocks = int(np.sum(pattern))

            # Fetch the nonzero rows and columns
            rows: List[int] = []
            cols: List[int] = []

            for row in range(pattern.shape[0]):
                for col in range(pattern.shape[1]):
                    if pattern[row, col] == 1:
                        rows.append(row)
                        cols.append(col)

            weights: List[tf.Variable] = []
            for idx in range(num_blocks):
                # Create the trainable weight matrix, [D, D]
                weight = tf.get_variable('kernel-{0}'.format(idx),
                                         shape=[block_size, block_size],
                                         dtype=inputs.dtype,
                                         initializer=tf.glorot_uniform_initializer(),
                                         trainable=True)
                weights.append(weight)

            # Transform the input
            transformed = block_sparse_matmul(dense_mat=inputs,
                                              blocks=weights,
                                              nonzero_rows=tf.constant(rows, dtype=tf.int32),
                                              nonzero_cols=tf.constant(cols, dtype=tf.int32),
                                              output_dims=units)

            # Transform the input
            #transformed = block_diagonal_matmul(dense_mat=inputs,
            #                                    blocks=weights)  # [B, M]

        # Apply the random connections to combine information between blocks 
        random_conn = tf.compat.v1.get_variable('random-conn',
                                                shape=[1, units],
                                                dtype=inputs.dtype,
                                                initializer=tf.compat.v1.glorot_uniform_initializer(),
                                                trainable=True)

        gathered = tf.gather(transformed, indices=sparse_indices, axis=-1)  # [B, M]
        sparse_transformed = tf.multiply(gathered, random_conn)
        
        transformed = tf.add(transformed, sparse_transformed)

        # Apply bias if required
        if use_bias:
            bias = tf.get_variable(name='bias',
                                             shape=(1, units),
                                             initializer=tf.random_uniform_initializer(minval=-0.7, maxval=0.7),
                                             trainable=True)

            transformed = tf.add(transformed, bias)

        # Apply the activation function
        activation_fn = get_activation(activation)

        if activation_fn is not None:
            output = activation_fn(transformed)
        else:
            output = transformed

        # Apply dropout if required
        if use_dropout:
            output = tf.nn.dropout(output, rate=1.0 - dropout_keep_rate)

        return output
