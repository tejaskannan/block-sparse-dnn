import tensorflow as tf

from .base import DenseNeuralNetwork
from utils.constants import INPUTS, OUTPUT, DROPOUT_KEEP_RATE, PREDICTION_OP
from utils.constants import INPUT_SHAPE, OUTPUT_SHAPE, LOGITS_OP
from layers.fully_connected import fully_connected
from layers.conv import conv2d


class CNN(DenseNeuralNetwork):

    def make_graph(self, is_train: bool, is_frozen: bool):
        dropout_keep_rate = self._placeholders[DROPOUT_KEEP_RATE]

        # Apply the convolutions
        state = self._placeholders[INPUTS]  # [B, H, W] or [B, H, W, C]

        conv_params = zip(self._hypers['filter_sizes'], self._hypers['filter_strides'], self._hypers['pool_strides'], self._hypers['out_channels'])

        for i, (filter_size, filter_stride, pool_stride, out_channels) in enumerate(conv_params):
            state = conv2d(inputs=state,
                           filter_size=filter_size,
                           filter_stride=filter_stride,
                           pool_stride=pool_stride,
                           out_channels=out_channels,
                           activation=self._hypers['conv_activation'],
                           dropout_keep_rate=dropout_keep_rate,
                           use_dropout=is_train,
                           pool_mode=self._hypers['conv_pool_mode'],
                           name='conv-{0}'.format(i))

        # Flatten the state
        _, height, width, channels = state.get_shape()
        flattened = tf.reshape(state, shape=(-1, height * width * channels))  # [B, K]

        # Apply the feed-forward layers
        hidden = flattened  # [B, K]
        for j, hidden_units in enumerate(self._hypers['hidden_units']):
            hidden = fully_connected(inputs=hidden,
                                     units=hidden_units,
                                     activation=self._hypers['hidden_activation'],
                                     dropout_keep_rate=dropout_keep_rate,
                                     use_bias=True,
                                     use_dropout=is_train,
                                     should_layer_normalize=self._hypers['should_layer_normalize'],
                                     name='hidden-{0}'.format(j))

        output = fully_connected(inputs=hidden,
                                 units=self._metadata[OUTPUT_SHAPE],
                                 activation=None,
                                 dropout_keep_rate=dropout_keep_rate,
                                 use_bias=True,
                                 use_dropout=False,
                                 should_layer_normalize=False,
                                 name='output')

        # Define the output operations
        self._ops[LOGITS_OP] = output
        self._ops[PREDICTION_OP] = tf.nn.softmax(output, axis=-1)
