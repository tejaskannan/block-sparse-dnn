import tensorflow as tf
import numpy as np
from typing import Dict

from .base import DenseNeuralNetwork
from dataset.dataset import Batch
from utils.constants import INPUTS, OUTPUT, DROPOUT_KEEP_RATE, PREDICTION_OP, LOSS_OP, MAX_INPUT
from utils.constants import INPUT_SHAPE, OUTPUT_SHAPE, LOGITS_OP, BIG_NUMBER, SEQ_LENGTH
from utils.tf_utils import masked_weighted_avg
from utils.data_utils import get_seq_length
from layers.fully_connected import fully_connected


def make_rnn_cell(state_size: int, name: str) -> tf.compat.v1.nn.rnn_cell.RNNCell:
    return tf.compat.v1.nn.rnn_cell.GRUCell(num_units=state_size,
                                            activation=tf.math.tanh,
                                            kernel_initializer=tf.compat.v1.glorot_uniform_initializer(),
                                            bias_initializer=tf.random_uniform_initializer(minval=-0.7, maxval=0.7),
                                            dtype=tf.float32,
                                            name=name)


class RNN(DenseNeuralNetwork):

    def make_placeholders(self, is_frozen: bool):
        super().make_placeholders(is_frozen=is_frozen)

        # Include the sequence length placeholder
        if not is_frozen:
            self._placeholders[SEQ_LENGTH] = tf.compat.v1.placeholder(shape=(None,),
                                                                      dtype=tf.int32,
                                                                      name=SEQ_LENGTH)
        else:
            self._placeholders[SEQ_LENGTH] = tf.ones(shape=(1,), dtype=tf.int32, name=SEQ_LENGTH)

    def batch_to_feed_dict(self, batch: Batch, is_train: bool) -> Dict[tf.compat.v1.placeholder, np.ndarray]:
        seq_length = get_seq_length(embeddings=batch.inputs)  # [B]

        return {
            self._placeholders[INPUTS]: batch.inputs,
            self._placeholders[OUTPUT]: batch.output,
            self._placeholders[SEQ_LENGTH]: seq_length,
            self._placeholders[DROPOUT_KEEP_RATE]: self._hypers[DROPOUT_KEEP_RATE] if is_train else 1.0
        }

    def make_graph(self, is_train: bool, is_frozen: bool):
        embeddings = self._placeholders[INPUTS]  # [B, T, D]

        # Make the (multi-layer) GRU Cell
        state_size = self._hypers['state_size']

        cells = [make_rnn_cell(state_size, name='cell-{0}'.format(i)) for i in range(self._hypers['rnn_layers'])]
        rnn_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells=cells)

        # Apply the RNN
        initial_state = rnn_cell.zero_state(batch_size=tf.shape(embeddings)[0],
                                            dtype=tf.float32)
        rnn_outputs, _  = tf.compat.v1.nn.dynamic_rnn(inputs=embeddings,
                                                      cell=rnn_cell,
                                                      initial_state=initial_state,
                                                      dtype=tf.float32,
                                                      scope='rnn')

        # Create the (self) attention weights, [B, T, 1]
        attn_weights = fully_connected(inputs=rnn_outputs,
                                       units=1,
                                       activation=None,
                                       dropout_keep_rate=1.0,
                                       use_bias=True,
                                       use_dropout=False,
                                       name='attn-weights')

        # Aggregate the RNN outputs, [B, D]
        rnn_result = masked_weighted_avg(inputs=rnn_outputs, 
                                         weights=attn_weights,
                                         seq_length=self._placeholders[SEQ_LENGTH])

        # Execute the output, feed-forward layers
        hidden = rnn_result  # [B, D]
        for hidden_idx, hidden_units in enumerate(self._hypers['hidden_units']):
            transformed = fully_connected(inputs=hidden,
                                          units=hidden_units,
                                          activation=self._hypers['hidden_activation'],
                                          dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                                          use_bias=True,
                                          use_dropout=is_train,
                                          name='hidden-{0}'.format(hidden_idx))
            hidden = transformed

        output_units = self._metadata[OUTPUT_SHAPE]
        logits = fully_connected(inputs=hidden,
                                 units=output_units,
                                 activation=None,
                                 dropout_keep_rate=1.0,
                                 use_bias=True,
                                 use_dropout=False,
                                 name='output')

        self._ops[LOGITS_OP] = logits
        self._ops[PREDICTION_OP] = tf.nn.softmax(logits, axis=-1)
