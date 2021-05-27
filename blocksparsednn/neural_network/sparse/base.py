import tensorflow as tf
import numpy as np
from collections import namedtuple
from typing import Dict, Any, List, Tuple

from ..base import NeuralNetwork
from dataset.dataset import Batch
from utils.constants import INPUTS, OUTPUT, DROPOUT_KEEP_RATE, PREDICTION_OP, LOSS_OP, SPARSE_NAMES, SMALL_NUMBER
from utils.constants import INPUT_SHAPE, OUTPUT_SHAPE, SCALER, LOGITS_OP, SPARSE_DIMS, SPARSE_INDICES
from utils.sparsity import get_num_nonzero
from layers.fully_connected import sparse_connected


HIDDEN_FMT = 'hidden_{0}'
INDICES_FMT = '{0}_indices'


class SparseNeuralNetwork(NeuralNetwork):

    def __init__(self, name: str, hypers: Dict[str, Any], log_device: bool = False):
        super().__init__(name, hypers, log_device)
    
        self._rand = np.random.RandomState(seed=1072)

        # Dictionary to track information about the sparse layers
        self._sparse_indices: Dict[str, np.ndarray] = dict()
        self._sparse_dims: Dict[str, Tuple[int, int]] = dict()
        self._sparse_name: Dict[str, str] = dict()  # Maps layer name to placeholder name

    def init_sparse_placeholders(self, input_units: int, is_frozen: bool):
        # Create the sparsity schedule
        units = [input_units] + self._hypers['hidden_units'] + [self._metadata[OUTPUT_SHAPE]]

        sparsity = self._hypers['sparsity']

        # Initialize the sparse hidden layers
        layer_idx = 0
        layer_units: List[int] = [input_units]

        for hidden_idx, hidden_units in enumerate(self._hypers['hidden_units']):
            layer_name = HIDDEN_FMT.format(hidden_idx)
            indices_name = INDICES_FMT.format(layer_name)

            num_nonzero = get_num_nonzero(in_units=layer_units[-1],
                                          out_units=hidden_units,
                                          sparsity=sparsity)

            self.init_sparse_layer(name=layer_name,
                                   ph_name=indices_name,
                                   input_units=layer_units[-1],
                                   output_units=hidden_units,
                                   num_nonzero=num_nonzero)
            layer_units.append(hidden_units)                           

            hidden_nonzero = len(self._sparse_indices[layer_name])
            layer_idx += 1

            if not is_frozen:
                self._placeholders[indices_name] = tf.placeholder(shape=(hidden_nonzero, 2),
                                                                            dtype=tf.int64,
                                                                            name=indices_name)
            else:
                self._placeholders[indices_name] = self._sparse_indices[layer_name]

    def init_sparse_layer(self, name: str, ph_name: str, input_units: int, output_units: int, num_nonzero: int):
        # Initialize to previously-set values if available
        if (SPARSE_INDICES in self._metadata and SPARSE_DIMS in self._metadata) and \
            (name in self._metadata[SPARSE_INDICES] and name in self._metadata[SPARSE_DIMS]):
                self._sparse_dims = self._metadata[SPARSE_DIMS]
                self._sparse_indices = self._metadata[SPARSE_INDICES]
                self._sparse_name = self._metadata[SPARSE_NAMES]
                return

        # Add the dimensions of the sparse layer
        self._sparse_dims[name] = (output_units, input_units)
        self._sparse_name[name] = ph_name

        # Generate the initial sparse indices in a uniformly random manner
        all_idx = np.arange(output_units * input_units)
        num_nonzero = min(num_nonzero, len(all_idx))
        selected_idx = np.sort(self._rand.choice(all_idx, size=num_nonzero, replace=False))

        mat_indices: List[Tuple[int, int]] = []

        for idx in selected_idx:
            row = int(idx / input_units)
            col = int(idx % input_units)

            mat_indices.append((row, col))

        self._sparse_indices[name] = np.array(mat_indices)

        self._metadata[SPARSE_INDICES] = self._sparse_indices
        self._metadata[SPARSE_DIMS] = self._sparse_dims
        self._metadata[SPARSE_NAMES] = self._sparse_name

    def post_epoch_step(self, epoch: int, has_ended: bool):

        # Fetch the trainable weights for sparse variables
        sparse_var_dict: Dict[str, tf.Variable] = dict()

        with self._sess.graph.as_default():
            trainable_vars = self.get_trainable_vars()

            for var in trainable_vars:
                for layer_name in self._sparse_indices.keys():
                    if '{0}/kernel'.format(layer_name) in var.name:
                        sparse_var_dict[layer_name] = var

            sparse_vars = self._sess.run(sparse_var_dict)

        updated_weight_dict: Dict[str, np.ndarray] = dict()

        warmup = max(self._hypers.get('warmup', 0), 1)
        sparsity = self._hypers['sparsity']

        for sparse_layer_name, weights in sparse_vars.items():
            dims = self._sparse_dims[sparse_layer_name]
            n, m = dims[0], dims[1]

            # Get the current number of nonzero connections using the present amount of sparsity
            current_nonzero = get_num_nonzero(in_units=n, out_units=m, sparsity=sparsity)

            # Get the number of indices to prune
            num_to_prune = int(self._hypers['prune_fraction'] * current_nonzero)
            abs_weights = np.abs(weights)

            smallest_idx = np.argpartition(abs_weights, num_to_prune)[:num_to_prune]

            # Generate indices of new connections. The total number of weights
            # always remains the same.
            idx_set = set(((idx[0], idx[1]) for i, idx in enumerate(self._sparse_indices[sparse_layer_name]) if i not in smallest_idx))
            remaining_idx = [i for i in range(n * m) if (int(i / m), int(i % m)) not in idx_set]

            new_idx = np.sort(self._rand.choice(remaining_idx, size=num_to_prune, replace=False))

            new_conn = [(int(idx / m), int(idx % m)) for idx in new_idx]
            existing_conn = self._sparse_indices[sparse_layer_name]

            # Merge the connections lists to maintain alignment AND sort the sparse indices
            # by row. The sorting is necessary for compatibility with later sparse operations
            new_indices: List[Tuple[int, int]] = []
            new_weights: List[float] = []
            i, j = 0, 0

            init_bound = np.sqrt(6 / (n + m))  # Glorot Uniform Init bound

            for _ in range(len(weights)):
                while j in smallest_idx:
                    j += 1

                if i >= len(new_conn):
                    w = weights[j]
                    index = existing_conn[j]
                    j += 1
                elif j >= len(existing_conn):
                    index = new_conn[i]
                    w = self._rand.uniform(low=-init_bound, high=init_bound) if not has_ended else 0.0
                    i += 1
                else:
                    row_less = bool(new_conn[i][0] < existing_conn[j][0])
                    row_eq = bool(new_conn[i][0] == existing_conn[j][0])
                    col_less = bool(new_conn[i][1] < existing_conn[j][1])
                    col_eq = bool(new_conn[i][1] == existing_conn[j][1])

                    assert not (row_eq and col_eq), 'Detected a conflict: {0}, {1}'.format(new_conn[i], existing_conn[j])

                    if row_less or (row_eq and col_less):
                        index = new_conn[i]
                        w = self._rand.uniform(low=-init_bound, high=init_bound) if not has_ended else 0.0
                        i += 1
                    else:
                        w = weights[j]
                        index = existing_conn[j]
                        j += 1

                new_indices.append(index)
                new_weights.append(w)

                if len(new_weights) == len(weights):
                    break

            updated_weight_dict[sparse_layer_name] = np.array(new_weights)
            self._sparse_indices[sparse_layer_name] = np.array(new_indices)

            num_nonzero = len(new_weights) - np.sum(np.isclose(new_weights, 0))

        # Execute the assignment
        with self._sess.graph.as_default():
            ops = {name: tf.assign(sparse_var_dict[name], updated_weight_dict[name]) for name in sparse_var_dict.keys()}
            self._sess.run(ops)

        # Save the sparse information in the meta-data dict
        self._metadata[SPARSE_INDICES] = self._sparse_indices
        self._metadata[SPARSE_DIMS] = self._sparse_dims
        self._metadata[SPARSE_NAMES] = self._sparse_name

    def batch_to_feed_dict(self, batch: Batch, is_train: bool) -> Dict[tf.placeholder, np.ndarray]:
        batch_samples = len(batch.inputs)

        if self._hypers['should_normalize_inputs']:
            normalized_inputs = self._metadata[SCALER].transform(batch.inputs.reshape(batch_samples, -1))
            normalized_inputs = normalized_inputs.reshape(batch.inputs.shape)
        else:
            normalized_inputs = batch.inputs

        feed_dict = {
            self._placeholders[INPUTS]: normalized_inputs,
            self._placeholders[OUTPUT]: batch.output,
            self._placeholders[DROPOUT_KEEP_RATE]: self._hypers[DROPOUT_KEEP_RATE] if is_train else 1.0
        }

        # Add the sparse layer indices
        for layer_name, indices in self._sparse_indices.items():
            ph_name = self._sparse_name[layer_name]

            feed_dict[self._placeholders[ph_name]] = indices

        return feed_dict

    def make_placeholders(self, is_frozen: bool):

        if not is_frozen:
            self._placeholders[INPUTS] = tf.placeholder(shape=(None,) + self._metadata[INPUT_SHAPE],
                                                                  dtype=tf.float32,
                                                                  name=INPUTS)
            self._placeholders[OUTPUT] = tf.placeholder(shape=(None),
                                                                  dtype=tf.int32,
                                                                  name=OUTPUT)
            self._placeholders[DROPOUT_KEEP_RATE] = tf.placeholder(shape=[],
                                                                             dtype=tf.float32,
                                                                             name=DROPOUT_KEEP_RATE)
        else:
            self._placeholders[INPUTS] = tf.ones(shape=(1,) + self._metadata[INPUT_SHAPE], dtype=tf.float32, name=INPUTS)
            self._placeholders[OUTPUT] = tf.ones(shape=(1), dtype=tf.int32, name=OUTPUT)
            self._placeholders[DROPOUT_KEEP_RATE] = 1.0

    def make_loss(self):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._ops[LOGITS_OP],
                                                                       labels=self._placeholders[OUTPUT])
        self._ops[LOSS_OP] = tf.reduce_mean(cross_entropy)
