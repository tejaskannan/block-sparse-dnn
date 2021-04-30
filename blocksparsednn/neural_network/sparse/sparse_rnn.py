import tensorflow as tf
import numpy as np
from typing import Dict, List

from dataset.dataset import Batch
from utils.constants import OUTPUT, INPUT_SHAPE, OUTPUT_SHAPE, DROPOUT_KEEP_RATE, BIG_NUMBER
from utils.constants import PREDICTION_OP, LOGITS_OP, INPUTS, MAX_INPUT, SEQ_LENGTH
from utils.tf_utils import masked_weighted_avg
from utils.data_utils import get_seq_length
from utils.sparsity import get_num_nonzero
from layers.sparse_rnn_cells import SparseGRUCell
from layers.fully_connected import sparse_connected, fully_connected
from .base import SparseNeuralNetwork, HIDDEN_FMT, INDICES_FMT


RNN_NAME = 'rnn'
GATES_NAME = 'rnn/multi_rnn_cell/cell_{0}/rnn-{0}/gate'
TRANSFORM_NAME = 'rnn/multi_rnn_cell/cell_{0}/rnn-{0}/transform'
OUTPUT_NAME = 'output'


class SparseRNN(SparseNeuralNetwork):

    def make_placeholders(self, is_frozen: bool):
        super().make_placeholders(is_frozen)

        # Include the sequence length placeholder
        if not is_frozen:
            self._placeholders[SEQ_LENGTH] = tf.compat.v1.placeholder(shape=(None,),
                                                                      dtype=tf.int32,
                                                                      name=SEQ_LENGTH)
        else:
            self._placeholders[SEQ_LENGTH] = tf.ones(shape=(1,), dtype=tf.int32, name=SEQ_LENGTH)

    def batch_to_feed_dict(self, batch: Batch, is_train: bool) -> Dict[tf.compat.v1.placeholder, np.ndarray]:
        feed_dict = super().batch_to_feed_dict(batch, is_train)

        # Get the 'true' sequence length for each sequence.
        seq_length = get_seq_length(embeddings=batch.inputs)
        feed_dict[self._placeholders[SEQ_LENGTH]] = seq_length
        return feed_dict

    def init_sparse_placeholders(self, input_units: int, is_frozen: bool):
        # Add the indices for the readout layers
        super().init_sparse_placeholders(input_units=self._hypers['state_size'],
                                         is_frozen=is_frozen)

        state_size = self._hypers['state_size']
        sparsity = self._hypers.get('start_sparsity', self._hypers['sparsity'])

        # Recurrent layer indices (for now, only works for GRU Cells)
        feature_size = input_units
        for i in range(self._hypers['rnn_layers']):
            gates_indices = INDICES_FMT.format(GATES_NAME.format(i))
            gates_mask = MASK_FMT.format(gates_indices)
            gates_nonzero = get_num_nonzero(in_units=state_size + feature_size,
                                            out_units=2 * state_size * 2,
                                            sparsity=sparsity)

            self.init_sparse_layer(name=GATES_NAME.format(i),
                                   ph_name=gates_indices,
                                   input_units=state_size + feature_size,
                                   output_units=2 * state_size,
                                   num_nonzero=gates_nonzero)

            transform_indices = INDICES_FMT.format(TRANSFORM_NAME.format(i))
            transform_mask = MASK_FMT.format(transform_indices)
            transform_nonzero = get_num_nonzero(in_units=state_size + feature_size,
                                                out_units=state_size,
                                                sparsity=sparsity)

            self.init_sparse_layer(name=TRANSFORM_NAME.format(i),
                                   ph_name=transform_indices,
                                   input_units=state_size + feature_size,
                                   output_units=state_size,
                                   num_nonzero=transform_nonzero)

            if not is_frozen:
                gates_nonzero = len(self._sparse_indices[GATES_NAME.format(i)])
                self._placeholders[gates_indices] = tf.compat.v1.placeholder(shape=(gates_nonzero, 2),
                                                                             dtype=tf.int64,
                                                                             name=gates_indices)
                self._placeholders[gates_mask] = tf.compat.v1.placeholder(shape=(gates_nonzero,),
                                                                          dtype=tf.float32,
                                                                          name=gates_mask)

                transform_nonzero = len(self._sparse_indices[TRANSFORM_NAME.format(i)])
                self._placeholders[transform_indices] = tf.compat.v1.placeholder(shape=(transform_nonzero, 2),
                                                                                 dtype=tf.int64,
                                                                                 name=transform_indices)
                self._placeholders[transform_mask] = tf.compat.v1.placeholder(shape=(transform_nonzero,),
                                                                              dtype=tf.float32,
                                                                              name=transform_mask)
            else:
                self._placeholders[gates_indices] = self._sparse_indices[GATES_NAME.format(i)]
                self._placeholders[gates_mask] = self._sparse_mask[GATES_NAME.format(i)]
                self._placeholders[transform_indices] = self._sparse_indices[TRANSFORM_NAME.format(i)]
                self._placeholders[transform_mask] = self._sparse_mask[TRANSFORM_NAME.format(i)]

            # After the initial layer, the input features come from previous RNN cells. Thus, the
            # feature size is now the state size.
            feature_size = state_size

    def make_graph(self, is_train: bool, is_frozen: bool):
        state_size = self._hypers['state_size']  # D
        dropout = self._placeholders[DROPOUT_KEEP_RATE]
        embeddings = self._placeholders[INPUTS]  # [B, T, D]
        embedding_size = self._metadata[INPUT_SHAPE][-1]  # D
        batch_size = tf.shape(embeddings)[0] if not is_frozen else embeddings.get_shape()[0]  # B

        # Initialize the sparse layers
        self.init_sparse_placeholders(input_units=embedding_size,
                                      is_frozen=is_frozen)

        # Make the Sparse RNN Cell
        cells: List[SparseGRUCell] = []
        for i in range(self._hypers['rnn_layers']):
            gate_indices = INDICES_FMT.format(GATES_NAME.format(i))
            gate_mask = MASK_FMT.format(gate_indices)

            transform_indices = INDICES_FMT.format(TRANSFORM_NAME.format(i))
            transform_mask = MASK_FMT.format(transform_indices)

            rnn_cell = SparseGRUCell(state_size=state_size,
                                     feature_size=self._metadata[INPUT_SHAPE][-1],
                                     gate_indices=self._placeholders[gate_indices],
                                     gate_mask=self._placeholders[gate_mask],
                                     transform_indices=self._placeholders[transform_indices],
                                     transform_mask=self._placeholders[transform_mask],
                                     name='{0}-{1}'.format(RNN_NAME, i))
            cells.append(rnn_cell)

        # Apply the RNN
        cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells=cells)

        init_state = cell.get_initial_state(batch_size=batch_size, inputs=embeddings, dtype=tf.float32)
        rnn_outputs, _ = tf.compat.v1.nn.dynamic_rnn(inputs=embeddings,
                                                     cell=cell,
                                                     initial_state=init_state,
                                                     dtype=tf.float32,
                                                     scope=RNN_NAME)

        # Get the (self) attention weights, [B, T, 1]. We use a dense layer
        # here because there are so few weights in this layer.
        attn_weights = fully_connected(inputs=rnn_outputs,
                                       units=1,
                                       activation='leaky_relu',
                                       dropout_keep_rate=1.0,
                                       use_bias=True,
                                       use_dropout=False,
                                       should_layer_normalize=False,
                                       name='attn-weights')

        # Aggregated the RNN outputs via a weighted average, [B, D]
        rnn_result = masked_weighted_avg(inputs=rnn_outputs,
                                         weights=attn_weights,
                                         seq_length=self._placeholders[SEQ_LENGTH])

        # Compute the output predictions via feed-forward layers
        hidden = rnn_result # [B, K]
        for hidden_idx, hidden_units in enumerate(self._hypers['hidden_units']):
            layer_name = HIDDEN_FMT.format(hidden_idx)
            indices_name = INDICES_FMT.format(layer_name)
            mask_name = MASK_FMT.format(indices_name)

            hidden = sparse_connected(inputs=hidden,
                                      units=hidden_units,
                                      activation=self._hypers['hidden_activation'],
                                      dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                                      use_bias=True,
                                      use_dropout=is_train,
                                      should_layer_normalize=self._hypers['should_layer_normalize'],
                                      weight_indices=self._placeholders[indices_name],
                                      weight_mask=self._placeholders[mask_name],
                                      name=layer_name)

        output_indices = INDICES_FMT.format(OUTPUT_NAME)
        output_mask = MASK_FMT.format(output_indices)
        logits = sparse_connected(inputs=hidden,
                                  units=self._metadata[OUTPUT_SHAPE],
                                  activation=None,
                                  dropout_keep_rate=1.0,
                                  use_bias=True,
                                  use_dropout=False,
                                  should_layer_normalize=False,
                                  weight_indices=self._placeholders[output_indices],
                                  weight_mask=self._placeholders[output_mask],
                                  name='output')

        self._ops[LOGITS_OP] = logits
        self._ops[PREDICTION_OP] = tf.nn.softmax(logits, axis=-1)  # [B, C]
