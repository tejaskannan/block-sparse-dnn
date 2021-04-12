import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, List, Union

from utils.tf_utils import get_activation, layer_normalize


def fully_connected(inputs: tf.Tensor,
                    units: int,
                    activation: Optional[str],
                    dropout_keep_rate: tf.Tensor,
                    use_bias: bool,
                    use_dropout: bool,
                    should_layer_normalize: bool,
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
        should_layer_normalize: Whether to apply layer normalization
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

        # Apply layer normalization (if needed)
        if should_layer_normalize:
            transformed = layer_normalize(transformed)  # [B, M]

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
                     should_layer_normalize: bool,
                     weight_indices: Union[tf.Tensor, np.ndarray],
                     weight_mask: Union[tf.Tensor, np.ndarray],
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
        should_layer_normalize: Whether to apply layer normalization
        weight_indices: A 2d tensor of (row, col) indices of the weight terms
        weight_mask: A 1d tensor containing a binary mask for the sparse weights
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

        # Mask out any weights to ensure zeros for already-zero indices
        weights = tf.math.multiply(weights, weight_mask)  # [K]

        weight_mat = tf.SparseTensor(indices=weight_indices,
                                     values=weights,
                                     dense_shape=dense_shape)  # [M, N]

        # Apply the weight matrix via a sparse matrix multiplication, [M, B]
        transp_transformed = tf.sparse.sparse_dense_matmul(weight_mat, inputs, adjoint_b=True)  # [M, B]
        transformed = tf.transpose(transp_transformed, perm=[1, 0])  # [B, M]

        # Apply layer normalization if specified
        if should_layer_normalize:
            transformed = layer_normalize(transformed)

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
