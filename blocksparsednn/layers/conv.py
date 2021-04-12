import tensorflow as tf
from typing import Optional

from utils.tf_utils import get_activation


def conv2d(inputs: tf.Tensor,
           filter_size: int,
           filter_stride: int,
           out_channels: int,
           pool_stride: int,
           dropout_keep_rate: tf.Tensor,
           use_dropout: bool,
           activation: Optional[str],
           pool_mode: Optional[str],
           name: str) -> tf.Tensor:
    """
    Executes a 2d convolution on the given inputs with a trainable filter.

    Args:
        inputs: A [B, H, W, C] or [B, W, H] input tensor. If no channels are given,
            then the number of channels is assumed to be one.
        filter_size: The size of the (square) convolution filter. Must be >= 1.
        filter_stride: The filter stride length. Must be >= 1.
        pool_stride: The pooling stride length. Must be >= 1.
        out_channels: The number of output channels (K). Must be >= 1.
        dropout_keep_rate: The dropout keep rate.
        use_dropout: Whether to apply dropout to the final state.
        activation: The optional activation function. If none, then no
            activation is applied.
        pool_mode: The pooling mode. Must be either 'avg', 'max' or None.
        name: The variable scope for this layer.
    Returns:
        A [B, H, W, K] transformed tensor.
    """
    assert filter_size >= 1, 'Must have a filter size of at least 1.'
    assert filter_stride >= 1, 'Must have a stride length of at least 1.'
    assert pool_stride >= 1, 'Must have a pooling stride of at least 1.'
    assert out_channels >= 1, 'Must have at least 1 output channel.'

    # Reshape inputs if necessary
    if len(inputs.get_shape()) == 3:
        inputs = tf.expand_dims(inputs, axis=-1)  # [B, H, W, C] where C = 1

    in_channels = inputs.get_shape()[-1]

    with tf.compat.v1.variable_scope(name):
        # Create the trainable filter
        filter_dims = [filter_size, filter_size, in_channels, out_channels]
        kernel = tf.compat.v1.get_variable(name='filter',
                                           shape=filter_dims,
                                           initializer=tf.compat.v1.random_uniform_initializer(minval=-0.7, maxval=0.7),
                                           dtype=tf.float32,
                                           trainable=True)

        # Apply the convolution filter, [B, H, W, C]
        conv = tf.nn.conv2d(input=inputs,
                            filters=kernel,
                            strides=filter_stride,
                            padding='SAME',
                            name='conv')

        # Apply the activation function
        activation_fn = get_activation(activation)
        if activation_fn is not None:
            conv = activation_fn(conv)

        # Apply the (optional) pooling layer
        if pool_mode is not None:        
            mode = pool_mode.lower()

            if mode in ('avg', 'average'):
                pooled = tf.nn.avg_pool2d(input=conv,
                                          ksize=(filter_size, filter_size),
                                          strides=pool_stride,
                                          padding='SAME',
                                          name='pool')
            elif mode in ('max', 'maximum'):
                pooled = tf.nn.max_pool2d(input=conv,
                                          ksize=(filter_size, filter_size),
                                          strides=pool_stride,
                                          padding='SAME',
                                          name='pool')
            elif mode == 'none':
                pooled = conv
            else:
                raise ValueError('Unknown pooling type: {0}'.format(pool_mode))
        else:
            pooled = conv  # No pooling

        # Apply the (optional) dropout layer
        if use_dropout:
            transformed = tf.nn.dropout(pooled, rate=1.0 - dropout_keep_rate)
        else:
            transformed = pooled

        return transformed


def residual_conv2d(inputs: tf.Tensor,
                    filter_size: int,
                    hidden_channels: int,
                    activation: Optional[str],
                    dropout_keep_rate: tf.Tensor,
                    pool_mode: Optional[str],
                    use_residual_conv: bool,
                    use_dropout: bool,
                    name: str):
    """
    Creates a residual block using two convolution layers.
    """
    # Get the number of input channels
    inpt_shape = inputs.get_shape()
    if len(inpt_shape) == 3:
        inputs = tf.expand_dims(inputs, axis=-1)  # [B, H, W, C] for C = 1

    in_channels = inputs.get_shape()[-1]

    # Apply the first convolution layer, [B, W, 
    hidden = conv2d(inputs=inputs,
                    filter_size=filter_size,
                    filter_stride=1,
                    out_channels=hidden_channels,
                    activation=activation,
                    pool_stride=1,
                    pool_mode=pool_mode,
                    dropout_keep_rate=dropout_keep_rate,
                    use_dropout=use_dropout,
                    name='{0}-1'.format(name))

    # Apply the second convolution layer. We avoid the activation function and apply it only after the residual connection.
    transformed = conv2d(inputs=hidden,
                         filter_size=filter_size,
                         filter_stride=1,
                         out_channels=in_channels,
                         activation=None,
                         pool_stride=1,
                         pool_mode=pool_mode,
                         dropout_keep_rate=dropout_keep_rate,
                         use_dropout=False,
                         name='{0}-2'.format(name))

    # Compute the residual value (either an identity or a separate 1 x 1 convolution)
    if use_residual_conv:
        residual = conv2d(inputs=inputs,
                          filter_size=1,
                          filter_stride=1,
                          out_channels=in_channels,
                          activation=None,
                          pool_stride=1,
                          pool_mode=None,
                          dropout_keep_rate=dropout_keep_rate,
                          use_dropout=False,
                          name='{0}-residual'.format(name))  # [B, H, W, C]
    else:
        residual = inputs  # [B, H, W, C]

    # Apply the residual connection, [B, H, W, C]
    res_transformed = transformed + residual

    # Apply the activation function
    activation_fn = get_activation(activation)
    if activation_fn is not None:
        res_transformed = activation_fn(res_transformed)

    # Apply dropout
    if use_dropout:
        res_transformed = tf.nn.dropout(res_transformed, rate=1.0 - dropout_keep_rate)

    return res_transformed
