import tensorflow as tf
import numpy as np
from typing import Dict, Any, List, Tuple

from .base import BlockSparseNeuralNetwork, HIDDEN_FMT, ROWS_FMT, COLS_FMT
from utils.constants import INPUTS, OUTPUT, DROPOUT_KEEP_RATE, PREDICTION_OP
from utils.constants import INPUT_SHAPE, OUTPUT_SHAPE, LOGITS_OP, SPARSE_DIMS, SPARSE_INDICES
from layers.fully_connected import block_sparse_connected, fully_connected



class BlockSparseMLP(BlockSparseNeuralNetwork):

    def make_graph(self, is_train: bool, is_frozen: bool):
        """
        Builds the Sparse MLP Graph.
        """
        # Initialize the sparse placeholders
        self.init_sparse_placeholders(input_units=self.num_input_features, is_frozen=is_frozen)

        self._ops['inputs'] = self._placeholders[INPUTS]

        hidden = self._placeholders[INPUTS]
        for hidden_idx, hidden_units in enumerate(self._hypers['hidden_units']):
            layer_name = HIDDEN_FMT.format(hidden_idx)
            rows_name = ROWS_FMT.format(layer_name)
            cols_name = COLS_FMT.format(layer_name)

            hidden = block_sparse_connected(inputs=hidden,
                                            units=hidden_units,
                                            activation=self._hypers['hidden_activation'],
                                            dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                                            use_bias=True,
                                            use_dropout=is_train,
                                            nonzero_rows=self._placeholders[rows_name],
                                            nonzero_cols=self._placeholders[cols_name],
                                            block_size=self.block_size,
                                            name=layer_name)

            self._ops['hidden-{0}'.format(hidden_idx)] = hidden


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
