import tensorflow as tf
import numpy as np
from typing import Dict, Any

from ..base import NeuralNetwork
from dataset.dataset import Batch
from utils.constants import INPUTS, OUTPUT, DROPOUT_KEEP_RATE, PREDICTION_OP, LOSS_OP
from utils.constants import INPUT_SHAPE, OUTPUT_SHAPE, SCALER, LOGITS_OP
from layers.fully_connected import fully_connected


INIT_LOSS_WEIGHT = 0.1
SPARSITY_LOSS_WEIGHT = 'sparsity_loss_weight'


class BlockMaskedNeuralNetwork(NeuralNetwork):

    def __init__(self, name: str, hypers: Dict[str, Any], log_device: bool = False):
        super().__init__(name=name,
                         hypers=hypers,
                         log_device=log_device)
        self._sparsity_loss_weight = INIT_LOSS_WEIGHT

    @property
    def block_size(self) -> int:
        return self._hypers['block_size']

    @property
    def sparsity(self) -> float:
        return self._hypers['sparsity']

    @property
    def sparsity_loss_weight(self) -> float:
        return self._hypers[SPARSITY_LOSS_WEIGHT]

    @property
    def warmup(self) -> int:
        return self._hypers['warmup']

    def make_graph(self, is_train: bool, is_frozen: bool):
        raise NotImplementedError()

    def post_epoch_step(self, epoch: int, has_ended: bool):
        if epoch < self.warmup:
            alpha = (1.0 / self.warmup) * np.log(self.sparsity_loss_weight / INIT_LOSS_WEIGHT)
            self._sparsity_loss_weight = min(INIT_LOSS_WEIGHT * np.exp(alpha * epoch), self.sparsity_loss_weight)
        else:
            self._sparsity_loss_weight = self.sparsity_loss_weight

        print(self._sparsity_loss_weight)

    def batch_to_feed_dict(self, batch: Batch, is_train: bool) -> Dict[tf.compat.v1.placeholder, np.ndarray]:
        batch_samples = len(batch.inputs)

        if self._hypers['should_normalize_inputs']:
            normalized_inputs = self._metadata[SCALER].transform(batch.inputs.reshape(batch_samples, -1))
            normalized_inputs = normalized_inputs.reshape(batch.inputs.shape)
        else:
            normalized_inputs = batch.inputs

        return {
            self._placeholders[INPUTS]: normalized_inputs,
            self._placeholders[OUTPUT]: batch.output,
            self._placeholders[DROPOUT_KEEP_RATE]: self._hypers[DROPOUT_KEEP_RATE] if is_train else 1.0,
            self._placeholders[SPARSITY_LOSS_WEIGHT]: self._sparsity_loss_weight
        }

    def make_placeholders(self, is_frozen: bool):
        if not is_frozen:
            self._placeholders[INPUTS] = tf.compat.v1.placeholder(shape=(None,) + self._metadata[INPUT_SHAPE],
                                                                  dtype=tf.float32,
                                                                  name=INPUTS)
            self._placeholders[OUTPUT] = tf.compat.v1.placeholder(shape=(None),
                                                                  dtype=tf.int32,
                                                                  name=OUTPUT)
            self._placeholders[DROPOUT_KEEP_RATE] = tf.compat.v1.placeholder(shape=[],
                                                                             dtype=tf.float32,
                                                                             name=DROPOUT_KEEP_RATE)
            self._placeholders[SPARSITY_LOSS_WEIGHT] = tf.compat.v1.placeholder(shape=[],
                                                                                dtype=tf.float32,
                                                                                name=SPARSITY_LOSS_WEIGHT)
        else:
            self._placeholders[INPUTS] = tf.ones(shape=(1, ) + self._metadata[INPUT_SHAPE], dtype=tf.float32, name=INPUTS)
            self._placeholders[OUTPUT] = tf.ones(shape=(1), dtype=tf.int32, name=OUTPUT)
            self._placeholders[DROPOUT_KEEP_RATE] = 1.0
            self._placeholders[SPARSITY_LOSS_WEIGHT] = self.sparsity_loss_weight

    def make_loss(self):
        # Compute the cross entropy loss
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._ops[LOGITS_OP],
                                                                       labels=self._placeholders[OUTPUT])

        # Get all the weight matrices
        var_dict = {var.name: var for var in self.get_trainable_vars() if 'kernel' in var.name}

        total_units = 0
        nonzero_units: List[Union[tf.Tensor, int]] = []

        for var_name in var_dict.keys():
            name_tokens = var_name.split('/')
            mask_name = '{0}/binary-mask'.format(''.join(name_tokens[0:-1]))

            if mask_name in self._ops:
                nonzero_blocks = tf.reduce_sum(self._ops[mask_name])
                nonzero_units.append(nonzero_blocks)
            else:
                nonzero_units.append(np.prod(var_dict[var_name].get_shape()))

            total_units += np.prod(var_dict[var_name].get_shape())

        used_fraction = tf.reduce_sum(nonzero_units) / total_units  # Scalar
        sparsity_loss = self._placeholders[SPARSITY_LOSS_WEIGHT] * tf.square(used_fraction - self.sparsity)  # Scalar

        self._ops[LOSS_OP] = tf.reduce_mean(cross_entropy) + sparsity_loss
