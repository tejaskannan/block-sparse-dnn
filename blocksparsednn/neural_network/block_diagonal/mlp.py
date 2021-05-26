import tensorflow as tf
import numpy as np

from .base import BlockDiagNeuralNetwork
from blocksparsednn.utils.constants import INPUTS, OUTPUT, DROPOUT_KEEP_RATE, PREDICTION_OP, LOSS_OP
from blocksparsednn.utils.constants import INPUT_SHAPE, OUTPUT_SHAPE, LOGITS_OP
from blocksparsednn.layers.fully_connected import block_diagonal_connected, fully_connected


class BlockDiagMLP(BlockDiagNeuralNetwork):

    def make_graph(self, is_train: bool, is_frozen: bool):
        # Set the initial hidden state to the inputs
        hidden = self._placeholders[INPUTS]
        in_units = self.num_input_features
        # in_units = self.input_shape[-1]

        rand = np.random.RandomState(seed=53879)

        # Apply the hidden layers
        for hidden_idx, hidden_units in enumerate(self._hypers['hidden_units']):
            random_conn = np.arange(hidden_units)
            rand.shuffle(random_conn)

            layer_name = 'hidden-{0}'.format(hidden_idx)

            random_conn_name = '{0}/random-conn-idx'.format(layer_name)
            self._metadata[random_conn_name] = random_conn

            transformed = block_diagonal_connected(inputs=hidden,
                                                   units=hidden_units,
                                                   in_units=in_units,
                                                   activation='relu',
                                                   dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                                                   use_bias=True,
                                                   use_dropout=is_train,
                                                   block_size=self.block_size,
                                                   sparse_indices=random_conn,
                                                   name=layer_name,
                                                   use_bsmm=self.use_bsmm,
                                                   use_shuffle=self.use_shuffle)
            
            hidden = transformed
            in_units = hidden_units

        # Get the output dimension size
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
