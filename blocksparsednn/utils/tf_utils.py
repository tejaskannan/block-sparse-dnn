import tensorflow as tf
from typing import Callable, Optional, Union, List

from .constants import SMALL_NUMBER, BIG_NUMBER


def get_activation(name: Optional[str]) -> Optional[Callable[[tf.Tensor], tf.Tensor]]:
    """
    Creates an activation function with the given name.
    """
    if name is None:
        return None

    name = name.lower()
    if name == 'linear':
        return None
    elif name == 'relu':
        return tf.nn.relu
    elif name == 'tanh':
        return tf.math.tanh
    elif name == 'sigmoid':
        return tf.math.sigmoid
    elif name == 'leaky_relu':
        return tf.nn.leaky_relu
    else:
        raise ValueError('Unknown activation function with name: {0}'.format(name))


def get_optimizer(name: str,
                  learning_rate: float,
                  decay_rate: float,
                  decay_steps: int,
                  global_step: tf.Tensor) -> tf.train.Optimizer:
    """
    Makes the optimizer with the given name and learning rate.
    """
    # Create a learning rate with exponential decay
    lr = tf.train.exponential_decay(learning_rate=learning_rate,
                                              decay_rate=decay_rate,
                                              decay_steps=decay_steps,
                                              global_step=global_step,
                                              staircase=True)
    name = name.lower()
    if name == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate=lr)
    elif name == 'ada_delta':
        return tf.train.AdaDeltaOptimizer(learning_rate=lr)
    elif name == 'adam':
        return tf.train.AdamOptimizer(learning_rate=lr)
    elif name == 'ada_grad':
        return tf.train.AdaGradOptimizer(learning_rate=lr)
    else:
        raise ValueError('Unknown optimizer with name: {0}'.format(name))


def layer_normalize(inputs: tf.Tensor) -> tf.Tensor:
    """
    Normalizes the activations within each layer to a standard normal distribution.
    """
    # Compute the moments over the final dimension
    ndims = len(inputs.get_shape())
    mean, var = tf.nn.moments(inputs, axes=-1, keepdims=True)
    std = tf.sqrt(var)

    return (inputs - mean) / (std + SMALL_NUMBER)


def batch_normalize(inputs: tf.Tensor) -> tf.Tensor:
    """
    Normalizes the given features to a standard normal distribution.

    Args:
        inputs: A [B, N] tensor of features (N) for each batch element (B)
    Returns:
        A [B, N] tensor containing the normalized features
    """
    # Compute the mean [B, 1] and variance [B, 1] over the batch
    mean, variance = tf.nn.moments(inputs, axes=[0], keepdims=True)
    return tf.nn.batch_normalization(inputs, mean=mean, variance=variance, shift=None, scale=None)


def masked_weighted_avg(inputs: tf.Tensor, weights: tf.Tensor, seq_length: tf.Tensor) -> tf.Tensor:
    """
    Computes the weighted average of the input features over the provided sequence length.

    Args:
        inputs: A [B, T, D] tensor of features (D) for each sequence element (T) and batch sample (B)
        weights: A [B, T, 1] tensor of un-normalized weights.
        seq_length: A [B] tensor containing the sequence length of each batch sample
    Returns:
        A [B, D] tensor containing the aggregated features (D) for each batch sample (B)
    """
    indices = tf.expand_dims(tf.range(start=0, limit=inputs.get_shape()[1]), axis=0)  # [1, T]

    expanded_length = tf.expand_dims(seq_length, axis=1)  # [B, 1]
    mask = tf.cast(indices < expanded_length, dtype=tf.float32)  # Binary mask (1 when within range, 0 otherwise)
    mask = (1.0 - tf.expand_dims(mask, axis=-1)) * -BIG_NUMBER  # [B, T, 1]

    # Normalize the weights
    normalized_weights = tf.nn.softmax(weights + mask, axis=1)  # [B, T, 1]

    # Compute the aggregate features
    return tf.reduce_sum(inputs * normalized_weights, axis=1)  # [B, D]


def get_final_state(states: tf.Tensor, seq_length: tf.Tensor) -> tf.Tensor:
    """
    Retrieves the final state from the states tensor using the given sequence length.

    Args:
        states: A [B, T, D] tensor containing the states for each seq element (T)
        seq_length: A [B] tensor containing the true sequence lengths for each batch sample
    """
    batch_idx = tf.expand_dims(tf.range(start=0, limit=tf.shape(states)[0]), axis=-1)  # [B, 1]
    seq_idx = tf.expand_dims(seq_length - 1, axis=-1)  # [B, 1]
    final_idx = tf.concat([batch_idx, seq_idx], axis=-1)  # [B, 2]
    return tf.gather_nd(states, indices=final_idx)  # [B, D]


def project_block_mask(block_mask: tf.Tensor, block_size: int) -> tf.Tensor:
    """
    Projects the block pattern mask to the size of the full dense matrix.

    Args:
        block_mask: A [N / K, M / K] binary mask representing a block sparsity pattern
        block_size: The block size (K)
    Returns:
        A [N, M] binary mask for the full dense matrix.
    """
    block_mask = tf.repeat(block_mask, repeats=block_size, axis=0)  # [N, M / K]
    return tf.repeat(block_mask, repeats=block_size, axis=1)  # [N, M]


def block_diagonal_matmul(dense_mat: tf.Tensor, blocks: List[tf.Tensor]) -> tf.Tensor:
    """
    Performs a  block-diagonal matrix multiplication.

    Args:
        dense_mat: A [N, M] dense matrix
        blocks: A List of L [D, K] blocks. We must have
            that M / L == D.
    Returns:
        A [N, K * L] output.
    """
    dim1 = dense_mat.get_shape()[-1]

    num_blocks = len(blocks)
    in_block_size, out_block_size = blocks[0].get_shape()

    assert ((dim1 / num_blocks) - in_block_size) < SMALL_NUMBER, 'Misaligned blocks. Got {0} sized splits. Expected: {1}'.format(dim1 / num_blocks, in_block_size)

    splits = tf.split(dense_mat, num_or_size_splits=num_blocks, axis=-1)  # List of [N, D] blocks
    
    mul_results: List[tf.Tensor] = []
    for split, block in zip(splits, blocks):
        mul = tf.matmul(split, block)  # [N, K]
        mul_results.append(mul)

    return tf.concat(mul_results, axis=-1)  # [N, L * K]


def block_sparse_matmul(dense_mat: tf.Tensor,
                        blocks: List[tf.Tensor],
                        nonzero_rows: tf.Tensor,
                        nonzero_cols: tf.Tensor,
                        output_dims: int) -> tf.Tensor:
    """
    Performs a block-sparse matrix multiplication.

    Args:
        dense_mat: A [N, M] dense matrix
        blocks: A List of L [D, D] blocks
        nonzero_rows: A [L] tensor containing the row of each block
        nonzero_cols: A [L] tensor containing the column of each block
        output_dims: The size of the output tensor (K)
    Returns:
        A [N, K] dense tensor containing the result of the matrix
        multiplication.
    """
    dim1 = dense_mat.get_shape()[-1]

    # Holds a list of the [D, N] results
    block_results: List[tf.Tensor] = []
    output_indices: List[tf.Tensor] = []

    # Perform the matrix multiplication for each block
    for idx, block in enumerate(blocks):
        block_size = blocks[0].get_shape()[0]

        # Ensure proper alignment
        assert (dim1 % block_size) == 0, 'Misaligned blocks. {0} not divisible by {1}'.format(dim1, block_size)

        start_idx = nonzero_rows[idx] * block_size
        dense_slice = tf.slice(dense_mat, begin=[0, start_idx], size=[-1, block_size])  # [N, D]

        mul = tf.matmul(dense_slice, block)  # [N, D]

        # block_indices = tf.range(start=0, limit=block_size)  # [D]
        # col_indices = block_indices + (nonzero_cols[idx] * block_size)
        right_pad = output_dims - (nonzero_cols[idx] + 1) * block_size
        left_pad = nonzero_cols[idx] * block_size

        # Pad up the output dimension
        mul = tf.pad(mul, [[0, 0], [left_pad, right_pad]])  # [N, K]

        block_results.append(tf.expand_dims(mul, axis=-1))
        # output_indices.append(col_indices)

    # [K, N] result array
    # segment_ids = tf.concat(output_indices, axis=0)  # [L * D]
    block_concat = tf.concat(block_results, axis=-1)  # [N, K, L]
    return tf.reduce_sum(block_concat, axis=-1)  # [N, K]

    #result = tf.math.unsorted_segment_sum(block_concat,
    #                                      segment_ids=segment_ids,
    #                                      num_segments=output_dims)

    #return tf.transpose(result, perm=[1, 0])


def upper_triangular_mask(n: tf.Tensor) -> tf.Tensor:
    """
    Creates an upper triangular [n, n] tensor with -BIG_NUMBER on
    in the upper elements and zero otherwise. The diagonal is zero.
    """
    idx = tf.range(start=0, limit=n)
    rows = tf.expand_dims(idx, axis=-1)  # [n, 1]
    cols = tf.expand_dims(idx, axis=0)  # [1, n]

    mat =  tf.cast(tf.less(rows, cols), dtype=tf.float32)  # [n, n]
    return mat * -BIG_NUMBER  # [n, n]


def scaled_attention(query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, dropout_keep_rate: tf.Tensor) -> tf.Tensor:
    """
    Computes the scaled, dot-product attention for the given query, key and value.

    Args:
        query: A [B, T, D] tensor containing the query features (D) for each sequence element (T) and batch sample (B)
        key: A [B, T, D] tensor containing the key features (D)
        value: A [B, T, D] tensor containing the value features (D)
        dropout_keep_rate: The dropout keep rate.
    Returns:
        A [B, T, D] tensor containing the (masked) aggregate features for each sample.
    """
    _, seq_length, d_k = query.get_shape()

    # Compute the (scaled) attention weights
    query_key_prod = tf.matmul(query, key, transpose_b=True)  # [B, T, T]
    query_key_prod = query_key_prod / tf.sqrt(tf.cast(d_k, dtype=tf.float32))  # [B, T, T]
    query_key_prod = tf.nn.dropout(query_key_prod, rate=1.0 - dropout_keep_rate)  # [B, T, T]

    # Create the masked attention weights
    mask = tf.expand_dims(upper_triangular_mask(n=seq_length), axis=0)  # [1, T, T]
    attn_weights = tf.nn.softmax(query_key_prod + mask, axis=-1)  # [B, T, T]

    return tf.matmul(attn_weights, value), attn_weights


def pos_encoding(dims: int, seq_length: int) -> tf.Tensor:
    """
    Computes position encoding vectors for the given sequence length and state size.

    Args:
        dims: The number of state dimensions (D)
        seq_length: The number of elements in the sequence (T)
    Returns:
        A [T, D] tensor containing the positional encoding values.
    """
    pos_idx = tf.expand_dims(tf.range(start=0, limit=seq_length, dtype=tf.float32), axis=-1)  # [T, 1]

    dim_idx = tf.expand_dims(tf.range(start=0, limit=dims, delta=2, dtype=tf.float32), axis=0)  # [1, D / 2]
    denominator = tf.exp(dim_idx * (-tf.math.log(10000.0) / dims))  # [1, D / 2]
    pos_factor = tf.repeat(denominator, repeats=2, axis=-1)  # [1, D]

    # For an odd number of dimensions, the pos_factor will have D + 1 elements.
    # We slice off the last one here.
    if dims % 2 == 1:
        pos_factor = pos_factor[:, :dims]  # [1, D]

    sin_mask = tf.cast(tf.equal(tf.math.mod(tf.range(start=0, limit=dims), 2), 0), dtype=tf.float32)  # [D]
    sin_mask = tf.expand_dims(sin_mask, axis=0)  # [1, D]
    cos_mask = 1.0 - sin_mask

    pos_encoding = sin_mask * tf.math.sin(pos_idx * pos_factor) + cos_mask * tf.math.cos(pos_idx * pos_factor)  # [T, D]
    return pos_encoding


def add_residual_and_norm(x: tf.Tensor, res: tf.Tensor, dropout_keep_rate: tf.Tensor) -> tf.Tensor:
    norm_x = layer_normalize(x)
    drop_x = tf.nn.dropout(norm_x, rate=1.0 - dropout_keep_rate)
    return drop_x + res
