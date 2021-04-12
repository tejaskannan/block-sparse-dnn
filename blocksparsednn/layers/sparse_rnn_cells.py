import tensorflow as tf
from typing import Any, Optional, Tuple, List

from .fully_connected import sparse_connected


class SparseGRUCell(tf.compat.v1.nn.rnn_cell.RNNCell):

    def __init__(self,
                 state_size: int,
                 feature_size: int,
                 gate_indices: tf.Tensor,
                 gate_mask: tf.Tensor,
                 transform_indices: tf.Tensor,
                 transform_mask: tf.Tensor,
                 name: str):
        self._state_size = state_size  # D
        self._feature_size = feature_size  # N
        self._name = name
        self._gate_indices = gate_indices
        self._gate_mask = gate_mask
        self._transform_indices = transform_indices
        self._transform_mask = transform_mask

    @property
    def state_size(self) -> int:
        return self._state_size

    @property
    def output_size(self) -> int:
        return self._state_size

    def get_initial_state(self, inputs: Optional[tf.Tensor], batch_size: Optional[int], dtype: Any) -> tf.Tensor:
        """
        Creates the initial, zero state for this cell.
        """
        assert batch_size is not None, 'Must provide a batch size'

        with tf.compat.v1.variable_scope(self._name):
            init_state = tf.compat.v1.get_variable(name='init-state',
                                                   initializer=tf.zeros_initializer(),
                                                   shape=[1, self.state_size],
                                                   dtype=dtype,
                                                   trainable=False)

        return tf.tile(init_state, multiples=(batch_size, 1))  # [B, D]

    def __call__(self, inputs: tf.Tensor, state: tf.Tensor, scope: Optional[str] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Applies the RNN Cell to the given inputs and state
        """
        scope = scope if scope is not None else self._name
        with tf.compat.v1.variable_scope(scope):
            concat = tf.concat([inputs, state], axis=-1)  # [B, N + D]

            # [B, 2 * D]
            gates = sparse_connected(inputs=concat,
                                     units=2 * self.state_size,
                                     activation='sigmoid',
                                     dropout_keep_rate=1.0,
                                     use_bias=True,
                                     use_dropout=False,
                                     should_layer_normalize=False,
                                     weight_indices=self._gate_indices,
                                     weight_mask=self._gate_mask,
                                     name='gate')

            # Pair of [B, D] arrays
            reset, update = tf.split(gates, num_or_size_splits=2, axis=-1)

            # Create the transformed state, [B, D]
            transform_inputs = tf.concat([inputs, state * reset], axis=-1)  # [B, N + D]
            transformed = sparse_connected(inputs=transform_inputs,
                                           units=self.state_size,
                                           activation='tanh',
                                           dropout_keep_rate=1.0,
                                           use_bias=True,
                                           use_dropout=False,
                                           should_layer_normalize=False,
                                           weight_indices=self._transform_indices,
                                           weight_mask=self._transform_mask,
                                           name='transform')

            next_state = update * transformed + (1.0 - update) * state  # [B, D]

            return next_state, next_state
