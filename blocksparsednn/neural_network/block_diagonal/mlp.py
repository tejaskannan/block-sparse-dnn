import tensorflow as tf

from .base import BlockDiagNeuralNetwork
from blocksparsednn.utils.constants import INPUTS, OUTPUT, DROPOUT_KEEP_RATE, PREDICTION_OP, LOSS_OP
from blocksparsednn.utils.constants import INPUT_SHAPE, OUTPUT_SHAPE, LOGITS_OP
from blocksparsednn.layers.fully_connected import block_diagonal_connected, fully_connected


class BlockDiagMLP(BlockDiagNeuralNetwork):

    def make_graph(self, is_train: bool, is_frozen: bool):
        # Set the initial hidden state to the inputs
        hidden = self._placeholders[INPUTS]

        # Apply the hidden layers
        for hidden_idx, hidden_units in enumerate(self._hypers['hidden_units']):
            transformed = block_diagonal_connected(inputs=hidden,
                                                   units=hidden_units,
                                                   activation=self._hypers['hidden_activation'],
                                                   dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                                                   use_bias=True,
                                                   use_dropout=is_train,
                                                   num_blocks=self.num_blocks,
                                                   should_layer_normalize=self._hypers['should_layer_normalize'],
                                                   name='hidden-{0}'.format(hidden_idx))
            hidden = transformed

        # Get the output dimension size
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
