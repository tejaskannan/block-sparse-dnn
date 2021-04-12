import tensorflow as tf

from .base import DenseNeuralNetwork
from utils.constants import INPUTS, OUTPUT, DROPOUT_KEEP_RATE, PREDICTION_OP, LOSS_OP
from utils.constants import INPUT_SHAPE, OUTPUT_SHAPE, LOGITS_OP
from layers.fully_connected import fully_connected


class MLP(DenseNeuralNetwork):

    def make_graph(self, is_train: bool, is_frozen: bool):
        # Execute the hidden layers
        hidden = self._placeholders[INPUTS]
        for hidden_idx, hidden_units in enumerate(self._hypers['hidden_units']):
            transformed = fully_connected(inputs=hidden,
                                          units=hidden_units,
                                          activation=self._hypers['hidden_activation'],
                                          dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                                          use_bias=True,
                                          use_dropout=is_train,
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
