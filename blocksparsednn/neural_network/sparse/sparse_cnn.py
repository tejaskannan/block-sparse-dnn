import tensorflow as tf
import numpy as np
from typing import Dict, Any, List, Tuple

from .base import SparseNeuralNetwork, HIDDEN_FMT, INDICES_FMT
from utils.constants import INPUTS, OUTPUT, DROPOUT_KEEP_RATE, PREDICTION_OP
from utils.constants import INPUT_SHAPE, OUTPUT_SHAPE, LOGITS_OP, SPARSE_DIMS, SPARSE_INDICES
from layers.fully_connected import sparse_connected
from layers.conv import conv2d


class SparseCNN(SparseNeuralNetwork):

    def make_graph(self, is_train: bool, is_frozen: bool):
        """
        Builds the Sparse CNN Graph.
        """
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
        units = height * width * channels
        flattened = tf.reshape(state, shape=(-1, units))  # [B, K]

        # Create the sparse indices placeholders for the feed-forward layers
        self.init_sparse_placeholders(input_units=units, is_frozen=is_frozen)

        # Apply the feed-forward layers to the flattened values
        hidden = flattened # [B, K]
        for hidden_idx, hidden_units in enumerate(self._hypers['hidden_units']):
            layer_name = HIDDEN_FMT.format(hidden_idx)
            indices_name = INDICES_FMT.format(layer_name)
            mask_name = MASK_FMT.format(indices_name)

            hidden = sparse_connected(inputs=hidden,
                                      units=hidden_units,
                                      activation=self._hypers['hidden_activation'],
                                      dropout_keep_rate=dropout_keep_rate,
                                      use_bias=True,
                                      use_dropout=is_train,
                                      should_layer_normalize=self._hypers['should_layer_normalize'],
                                      weight_indices=self._placeholders[indices_name],
                                      name=layer_name)

        output_units = self._metadata[OUTPUT_SHAPE]
        output_indices = INDICES_FMT.format('output')
        output_mask = MASK_FMT.format(output_indices)
        logits = sparse_connected(inputs=hidden,
                                  units=output_units,
                                  activation=None,
                                  dropout_keep_rate=dropout_keep_rate,
                                  use_bias=True,
                                  use_dropout=False,
                                  should_layer_normalize=False,
                                  weight_indices=self._placeholders[output_indices],
                                  name='output')

        self._ops[LOGITS_OP] = logits
        self._ops[PREDICTION_OP] = tf.nn.softmax(logits, axis=-1)
