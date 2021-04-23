import tensorflow as tf

from .base import BlockDiagNeuralNetwork
from blocksparsednn.utils.constants import INPUTS, OUTPUT, DROPOUT_KEEP_RATE, PREDICTION_OP, LOSS_OP
from blocksparsednn.utils.constants import INPUT_SHAPE, OUTPUT_SHAPE, LOGITS_OP
from blocksparsednn.layers.fully_connected import block_diagonal_connected, fully_connected


class BlockDiagMLP(BlockDiagNeuralNetwork):

    def make_graph(self, is_train: bool, is_frozen: bool):

        # Make the initial projection
        hidden = fully_connected(inputs=self._placeholders[INPUTS],
                                 units=self._hypers['hidden_units'][0],
                                 activation=self._hypers['hidden_activation'],
                                 dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                                 use_bias=True,
                                 use_dropout=is_train,
                                 should_layer_normalize=self._hypers['should_layer_normalize'],
                                 name='hidden-0')

        # Apply the hidden layers
        for hidden_idx, hidden_units in enumerate(self._hypers['hidden_units']):
            if hidden_idx == 0:
                continue
                
            transformed = block_diagonal_connected(inputs=hidden,
                                                   units=hidden_units,
                                                   activation=self._hypers['hidden_activation'],
                                                   dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                                                   use_bias=True,
                                                   use_dropout=is_train,
                                                   block_size=self.block_size,
                                                   should_layer_normalize=self._hypers['should_layer_normalize'],
                                                   name='hidden-{0}'.format(hidden_idx))
            hidden = transformed

        output_units = self._metadata[OUTPUT_SHAPE]
        logits = fully_connected(inputs=hidden,
                                 units=output_units,
                                 activation=None,
                                 dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                                 use_bias=True,
                                 use_dropout=False,
                                 should_layer_normalize=False,
                                 name='output')

        self._ops[LOGITS_OP] = logits
        self._ops[PREDICTION_OP] = tf.nn.softmax(logits, axis=-1)
