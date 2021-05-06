import tensorflow as tf
import numpy as np
from typing import Dict, Any, List, Tuple

from .base import SparseNeuralNetwork, HIDDEN_FMT, INDICES_FMT
from utils.constants import INPUTS, OUTPUT, DROPOUT_KEEP_RATE, PREDICTION_OP
from utils.constants import INPUT_SHAPE, OUTPUT_SHAPE, LOGITS_OP, SPARSE_DIMS, SPARSE_INDICES
from layers.fully_connected import sparse_connected, fully_connected



class SparseMLP(SparseNeuralNetwork):

    def make_graph(self, is_train: bool, is_frozen: bool):
        """
        Builds the Sparse MLP Graph.
        """
        # Initialize the sparse placeholders
        self.init_sparse_placeholders(input_units=self._metadata[INPUT_SHAPE][-1], is_frozen=is_frozen)

        hidden = self._placeholders[INPUTS]
        for hidden_idx, hidden_units in enumerate(self._hypers['hidden_units']):
            layer_name = HIDDEN_FMT.format(hidden_idx)
            indices_name = INDICES_FMT.format(layer_name)

            hidden = sparse_connected(inputs=hidden,
                                      units=hidden_units,
                                      activation='relu',
                                      dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                                      use_bias=True,
                                      use_dropout=is_train,
                                      weight_indices=self._placeholders[indices_name],
                                      name=layer_name)

        output_units = self._metadata[OUTPUT_SHAPE]
        logits = fully_connected(inputs=hidden,
                                 units=output_units,
                                 activation=None,
                                 dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                                 use_bias=True,
                                 use_dropout=False,
                                 name='output')

        self._ops[LOGITS_OP] = logits
        self._ops[PREDICTION_OP] = tf.nn.softmax(logits, axis=-1)
