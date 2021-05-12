import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, List, Union

from utils.tf_utils import get_activation, project_block_mask, block_diagonal_matmul, block_sparse_matmul


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
    with tf.compat.v1.variable_scope(name):
        # Make the trainable weight matrix
        input_units = inputs.get_shape()[-1]
        W = tf.compat.v1.get_variable(name='kernel',
                                      shape=(input_units, units),  # [N, M]
                                      initializer=tf.compat.v1.glorot_uniform_initializer(),
                                      trainable=True)

        transformed = tf.matmul(inputs, W)  # [B, M]

        # Apply bias if required
        if use_bias:
            bias = tf.compat.v1.get_variable(name='bias',
                                             shape=(1, units),
                                             initializer=tf.compat.v1.random_uniform_initializer(minval=-0.7, maxval=0.7),
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
    with tf.compat.v1.variable_scope(name):
        # Create the (sparse) weight matrix with dimensions [M, N]
        num_nonzero = weight_indices.get_shape()[0] if isinstance(weight_indices, tf.Tensor) else weight_indices.shape[0]  # K
        weights = tf.compat.v1.get_variable(name='kernel',
                                            shape=(num_nonzero, ),
                                            initializer=tf.compat.v1.glorot_uniform_initializer(),
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
            bias = tf.compat.v1.get_variable(name='bias',
                                             shape=(1, units),
                                             initializer=tf.compat.v1.random_uniform_initializer(minval=-0.7, maxval=0.7),
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
    with tf.compat.v1.variable_scope(name):
        # Make the trainable weight matrix
        input_units = inputs.get_shape()[-1]
        W = tf.compat.v1.get_variable(name='kernel',
                                      shape=(input_units, units),  # [N, M]
                                      initializer=tf.compat.v1.glorot_uniform_initializer(),
                                      dtype=inputs.dtype,
                                      trainable=True)

        # Make the trainable block mask
        block_pattern = tf.compat.v1.get_variable(name='block-mask',
                                                  shape=(input_units // block_size, units // block_size),
                                                  initializer=tf.compat.v1.glorot_uniform_initializer(),
                                                  dtype=inputs.dtype,
                                                  trainable=True)  # [N / K, M / K]

        # Turn the block pattern into a binary block mask
        block_mask = binary_round(tf.math.sigmoid(block_pattern), sparsity)  # [N / K, M / K]

        # Project the block mask up to the size of the weight variable, [N, M]
        block_mask = project_block_mask(block_mask, block_size=block_size)

        masked_weights = tf.multiply(W, block_mask)
        transformed = tf.matmul(inputs, masked_weights)  # [B, M]
        # transformed = tf.matmul(inputs, W)

        # Apply bias if required
        if use_bias:
            bias = tf.compat.v1.get_variable(name='bias',
                                             shape=(1, units),
                                             initializer=tf.compat.v1.random_uniform_initializer(minval=-0.7, maxval=0.7),
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

    with tf.compat.v1.variable_scope(name):
        
        weights: List[tf.Variable] = []
        for idx in range(num_blocks):
            weight = tf.compat.v1.get_variable('kernel-{0}'.format(idx),
                                               shape=[block_size, block_size],
                                               dtype=inputs.dtype,
                                               initializer=tf.compat.v1.glorot_uniform_initializer(),
                                               trainable=True)
            weights.append(weight)

        # Transform the input, [B, M]
        transformed = block_sparse_matmul(dense_mat=inputs,
                                          blocks=weights,
                                          nonzero_rows=nonzero_rows,
                                          nonzero_cols=nonzero_cols,
                                          output_dims=units)

        # Apply bias if required
        if use_bias:
            bias = tf.compat.v1.get_variable(name='bias',
                                             shape=(1, units),
                                             initializer=tf.compat.v1.random_uniform_initializer(minval=-0.7, maxval=0.7),
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
                             activation: Optional[str],
                             dropout_keep_rate: tf.Tensor,
                             use_bias: bool,
                             use_dropout: bool,
                             num_blocks: int,
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
        num_blocks: The number of blocks. Must be a divisor of N and M.
        name: The name prefix of this layer
    Returns:
        The transformed tensor, [B, M]
    """
    in_units = inputs.get_shape()[-1]

    assert in_units > 0, 'Must provide a positive number of units.'
    assert units > 0, 'Must provide a positive number of output units.'

    assert (in_units % num_blocks) == 0, '# Blocks ({0}) must divide the input units ({1}).'.format(num_blocks, in_units)
    assert (units % num_blocks) == 0, '# Blocks ({0}) must divide the output units ({1}).'.format(num_blocks, units)

    with tf.compat.v1.variable_scope(name):
        ops: List[tf.LinearOperatorFullMatrix] = []

        in_block_size = int(in_units / num_blocks)
        out_block_size = int(units / num_blocks)

        splits = tf.split(inputs, num_or_size_splits=num_blocks, axis=-1)

        weights: List[tf.Variable] = []
        for idx in range(num_blocks):
            # Create the trainable weight matrix, [D, D]
            weight = tf.compat.v1.get_variable('kernel-{0}'.format(idx),
                                               shape=[in_block_size, out_block_size],
                                               dtype=inputs.dtype,
                                               initializer=tf.compat.v1.glorot_uniform_initializer(),
                                               trainable=True)
            weights.append(weight)

        # Transform the input
        transformed = block_diagonal_matmul(dense_mat=inputs,
                                            blocks=weights)  # [B, M]

        # Apply bias if required
        if use_bias:
            bias = tf.compat.v1.get_variable(name='bias',
                                             shape=(1, units),
                                             initializer=tf.compat.v1.random_uniform_initializer(minval=-0.7, maxval=0.7),
                                             trainable=True)

            transformed = tf.add(transformed, bias)

        # Apply the activation function
        activation_fn = get_activation(activation)

        if activation_fn is not None:
            output = activation_fn(transformed)
        else:
            output = transformed

        # Combine information across disjoint blocks via a pairwise average
        # rolled = tf.roll(transformed, shift=out_block_size, axis=-1)  # [B, M]
        # transformed = tf.add(transformed, rolled) * 0.5

        # Apply dropout if required
        if use_dropout:
            output = tf.nn.dropout(output, rate=1.0 - dropout_keep_rate)

        return output
