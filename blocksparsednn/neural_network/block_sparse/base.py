import tensorflow as tf

import numpy as np
import math
from typing import Dict, List, Any

from blocksparsednn.neural_network.base import NeuralNetwork
from blocksparsednn.dataset.dataset import Batch
from blocksparsednn.utils.sparsity import get_num_nonzero
from blocksparsednn.utils.constants import INPUTS, OUTPUT, DROPOUT_KEEP_RATE, PREDICTION_OP, LOSS_OP
from blocksparsednn.utils.constants import INPUT_SHAPE, OUTPUT_SHAPE, SCALER, LOGITS_OP


HIDDEN_FMT = 'hidden-{0}'
BLOCK_ROWS = 'block-rows'
BLOCK_COLS = 'block-cols'
ROWS_FMT = '{0}-rows'
COLS_FMT = '{0}-cols'


class BlockSparseNeuralNetwork(NeuralNetwork):

    def __init__(self, name: str, hypers: Dict[str, Any], log_device: bool = False):
        super().__init__(name, hypers, log_device)
    
        self._rand = np.random.RandomState(seed=1072)
        self._threshold = 0.15
        self._min_threshold = 0.01
        self._start_threshold = 0.15

        # Dictionary to track information about the sparse layers
        self._rows: Dict[str, np.ndarray] = dict()
        self._cols: Dict[str, np.ndarray] = dict()
        self._sparse_name: Dict[str, str] = dict()  # Maps layer name to placeholder name

    @property
    def block_size(self) -> int:
        return self._hypers['block_size']

    @property
    def sparsity(self) -> float:
        return self._hypers['sparsity']

    @property
    def prune_fraction(self) -> float:
        return self._hypers['prune_fraction']

    @property
    def warmup(self) -> int:
        return self._hypers['warmup']

    @property
    def l2(self) -> float:
        return self._hypers['l2']

    @property
    def num_input_features(self) -> int:
        return int(math.ceil(self._metadata[INPUT_SHAPE][-1] / self.block_size) * self.block_size)

    def make_graph(self, is_train: bool, is_frozen: bool):
        raise NotImplementedError()

    def post_epoch_step(self, epoch: int, has_ended: bool):
        layer_units: List[int] = [self.num_input_features]

        if has_ended:
            return

        if epoch < self.warmup:
            alpha = (1.0 / self.warmup) * np.log(self._start_threshold / self._min_threshold)
            self._threshold = max(self._start_threshold * np.exp(-1 * alpha * epoch), self._min_threshold)
        else:
            self._threshold = self._min_threshold

        for hidden_idx, hidden_units in enumerate(self._hypers['hidden_units']):
            layer_name = HIDDEN_FMT.format(hidden_idx)
            self.evolve_layer(name=layer_name,
                              in_units=layer_units[-1],
                              out_units=hidden_units)

            layer_units.append(hidden_units)

    def evolve_layer(self, name: str, in_units: int, out_units: int):
        # Fetch the trainable weights for sparse variables
        sparse_var_dict: Dict[str, tf.Variable] = dict()

        with self._sess.graph.as_default():
            trainable_vars = self.get_trainable_vars()

            for var in trainable_vars:
                if '{0}/kernel'.format(name) in var.name:
                    sparse_var_dict[var.name] = var

            sparse_vars = self._sess.run(sparse_var_dict)

        updated_weight_dict: Dict[str, np.ndarray] = dict()

        # Get the current number of nonzero connections using the present amount of sparsity
        in_blocks = int(in_units / self.block_size)
        out_blocks = int(out_units / self.block_size)

        current_nonzero = get_num_nonzero(in_units=in_blocks,
                                          out_units=out_blocks,
                                          sparsity=self.sparsity)

        if current_nonzero == (in_blocks * out_blocks):
            return

        # Get the number of indices to prune
        # num_to_prune = max(int(self.prune_fraction * current_nonzero), 1)

        weights_list: List[np.ndarray] = []

        num_below_threshold = 0

        max_weights: List[float] = []
        for var_name, weights in sorted(sparse_vars.items()):
            max_weight = np.max(np.abs(weights))

            if max_weight < self._threshold:
                num_below_threshold += 1

            max_weights.append(max_weight)
            weights_list.append(weights)

        #num_to_prune = min(num_to_prune, num_below_threshold)
        num_to_prune = num_below_threshold

        if num_to_prune == 0:
            return

        # Get the index of the groups to prune
        smallest = np.argpartition(max_weights, num_to_prune)[:num_to_prune]

        # Get the set of indices of the possible connections to add
        existing_set = set(zip(self._rows[name], self._cols[name]))

        candidate_idx: Set[int] = set()
        existing_idx: List[int] = []

        for i in range(in_blocks * out_blocks):
            row = int(i % in_blocks)
            col = int(i / in_blocks)

            if (row, col) not in existing_set:
                candidate_idx.add(i)
            else:
                existing_idx.append(i)

        kept_idx = set(idx for i, idx in enumerate(existing_idx) if i not in smallest)
        pruned_idx = set(idx for i, idx in enumerate(existing_idx) if i in smallest)

        # Select the connections to add
        idx_to_add = self._rand.choice(list(candidate_idx), size=len(pruned_idx), replace=False)

        init_bound = np.sqrt(6 / (in_units + out_units))  # Glorot uniform bound

        prev_weight_idx = 0  # Tracks position in the existing weight list

        updated_weights: List[np.ndarray] = []
        updated_rows: List[int] = []
        updated_cols: List[int] = []

        for i in range(in_blocks * out_blocks):
            row = int(i % in_blocks)
            col = int(i / in_blocks)

            if (i in kept_idx):
                updated_weights.append(np.copy(weights_list[prev_weight_idx]))
                updated_rows.append(row)
                updated_cols.append(col)

                prev_weight_idx += 1
            elif (i in idx_to_add):
                new_weight = self._rand.uniform(low=-init_bound,
                                                high=init_bound,
                                                size=(self.block_size, self.block_size))

                updated_weights.append(new_weight)
                updated_rows.append(row)
                updated_cols.append(col)
            elif (i in pruned_idx):
                prev_weight_idx += 1

        # Execute the assignment
        with self._sess.graph.as_default():
            ops = {name: tf.assign(sparse_var_dict[name], updated_weights[i]) for i, name in enumerate(sorted(sparse_var_dict.keys()))}
            self._sess.run(ops)

        # Save the sparse information in the meta-data dict
        self._rows[name] = updated_rows
        self._cols[name] = updated_cols

        self._metadata[BLOCK_ROWS] = self._rows
        self._metadata[BLOCK_COLS] = self._cols

    def init_sparse_placeholders(self, input_units: int, is_frozen: bool):
        # Create the sparsity schedule
        units = [input_units] + self._hypers['hidden_units'] + [self._metadata[OUTPUT_SHAPE]]

        sparsity = self._hypers['sparsity']

        # Initialize the sparse hidden layers
        layer_units: List[int] = [input_units]

        for hidden_idx, hidden_units in enumerate(self._hypers['hidden_units']):
            layer_name = HIDDEN_FMT.format(hidden_idx)
            rows_name = ROWS_FMT.format(layer_name)
            cols_name = COLS_FMT.format(layer_name)

            in_blocks = int(layer_units[-1] / self.block_size)
            out_blocks = int(hidden_units / self.block_size)

            num_nonzero = get_num_nonzero(in_units=in_blocks,
                                          out_units=out_blocks,
                                          sparsity=sparsity)

            # num_nonzero = int(round(in_blocks * out_blocks * sparsity))

            self.init_sparse_layer(name=layer_name,
                                   input_units=in_blocks,
                                   output_units=out_blocks,
                                   num_nonzero=num_nonzero)
            layer_units.append(hidden_units)

            if not is_frozen:
                self._placeholders[rows_name] = tf.placeholder(shape=(num_nonzero,),
                                                                         dtype=tf.int32,
                                                                         name=rows_name)
                self._placeholders[cols_name] = tf.placeholder(shape=(num_nonzero,),
                                                                         dtype=tf.int32,
                                                                         name=cols_name)
            else:
                self._placeholders[rows_name] = self._rows[layer_name]
                self._placeholders[cols_name] = self._cols[layer_name]

    def init_sparse_layer(self, name: str, input_units: int, output_units: int, num_nonzero: int):
        # Initialize to previously-set values if available
        if (BLOCK_ROWS in self._metadata and BLOCK_COLS in self._metadata) and \
            (name in self._metadata[BLOCK_ROWS] and name in self._metadata[BLOCK_COLS]):
            self._rows = self._metadata[BLOCK_ROWS]
            self._cols = self._metadata[BLOCK_COLS]
            return

        # Generate the initial sparse indices in a uniformly random manner
        all_idx = np.arange(output_units * input_units)
        num_nonzero = min(num_nonzero, len(all_idx))

        rows: List[int] = []
        cols: List[int] = []

        rand_idx = [idx for idx in all_idx if (int(idx % input_units) != int(idx / input_units))]

        num_diag = len(all_idx) - len(rand_idx)
        selected_idx = set(self._rand.choice(rand_idx, size=max(num_nonzero - num_diag, 0), replace=False))

        for idx in all_idx:
            row = int(idx % input_units)
            col = int(idx / input_units)

            if (row == col):
                rows.append(row)
                cols.append(col)
            elif idx in selected_idx:
                rows.append(row)
                cols.append(col)

            if len(rows) >= num_nonzero:
                break

        #selected_idx = np.sort(self._rand.choice(all_idx, size=num_nonzero, replace=False))

        #rows: List[int] = []
        #cols: List[int] = []

        #for idx in selected_idx:
        #    row = int(idx % input_units)
        #    rows.append(row)

        #    col = int(idx / input_units)
        #    cols.append(col)

        self._rows[name] = rows
        self._cols[name] = cols

        self._metadata[BLOCK_ROWS] = self._rows
        self._metadata[BLOCK_COLS] = self._cols

    def batch_to_feed_dict(self, batch: Batch, is_train: bool) -> Dict[tf.placeholder, np.ndarray]:
        batch_samples = len(batch.inputs)

        if self._hypers['should_normalize_inputs']:
            normalized_inputs = self._metadata[SCALER].transform(batch.inputs.reshape(batch_samples, -1))
            normalized_inputs = normalized_inputs.reshape(batch.inputs.shape)
        else:
            normalized_inputs = batch.inputs

        # Pad the input to ensure it is a multiple of the block size
        pad_amt = self.num_input_features - normalized_inputs.shape[1]
        normalized_inputs = np.pad(normalized_inputs, [(0, 0), (0, pad_amt)], mode='constant', constant_values=0)

        # Create feed dictionary for inputs and outputs
        feed_dict = {
            self._placeholders[INPUTS]: normalized_inputs,
            self._placeholders[OUTPUT]: batch.output,
            self._placeholders[DROPOUT_KEEP_RATE]: self._hypers[DROPOUT_KEEP_RATE] if is_train else 1.0
        }

        # Include the block rows and columns
        for layer_name in self._rows.keys():
            rows_name = ROWS_FMT.format(layer_name)
            feed_dict[self._placeholders[rows_name]] = self._rows[layer_name]

            cols_name = COLS_FMT.format(layer_name)
            feed_dict[self._placeholders[cols_name]] = self._cols[layer_name]

        return feed_dict

    def make_placeholders(self, is_frozen: bool):
        input_units = self.num_input_features

        if not is_frozen:
            self._placeholders[INPUTS] = tf.placeholder(shape=(None,) + self._metadata[INPUT_SHAPE][:-1] + (input_units,),
                                                                  dtype=tf.float32,
                                                                  name=INPUTS)
            self._placeholders[OUTPUT] = tf.placeholder(shape=(None),
                                                                  dtype=tf.int32,
                                                                  name=OUTPUT)
            self._placeholders[DROPOUT_KEEP_RATE] = tf.placeholder(shape=[],
                                                                             dtype=tf.float32,
                                                                             name=DROPOUT_KEEP_RATE)
        else:
            self._placeholders[INPUTS] = tf.ones(shape=(1, ) + self._metadata[INPUT_SHAPE][:-1] + (input_units,), dtype=tf.float32, name=INPUTS)
            self._placeholders[OUTPUT] = tf.ones(shape=(1), dtype=tf.int32, name=OUTPUT)
            self._placeholders[DROPOUT_KEEP_RATE] = 1.0

        self.init_sparse_placeholders(input_units=input_units, is_frozen=is_frozen)

    def make_loss(self):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._ops[LOGITS_OP],
                                                                       labels=self._placeholders[OUTPUT])


        # Add regularization
        weights = [weight for weight in self.get_trainable_vars() if 'kernel' in weight.name]
        norms = tf.reduce_sum([tf.linalg.norm(weight) for weight in weights])
        
        self._ops[LOSS_OP] = tf.reduce_mean(cross_entropy) + self.l2 * norms
