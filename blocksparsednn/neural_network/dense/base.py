import tensorflow as tf
import numpy as np
from typing import Dict

from ..base import NeuralNetwork
from dataset.dataset import Batch
from utils.constants import INPUTS, OUTPUT, DROPOUT_KEEP_RATE, PREDICTION_OP, LOSS_OP
from utils.constants import INPUT_SHAPE, OUTPUT_SHAPE, SCALER, LOGITS_OP
from layers.fully_connected import fully_connected


class DenseNeuralNetwork(NeuralNetwork):

    def make_graph(self, is_train: bool, is_frozen: bool):
        raise NotImplementedError()

    def batch_to_feed_dict(self, batch: Batch, is_train: bool) -> Dict[tf.placeholder, np.ndarray]:
        batch_samples = len(batch.inputs)

        if self._hypers['should_normalize_inputs']:
            normalized_inputs = self._metadata[SCALER].transform(batch.inputs.reshape(batch_samples, -1))
            normalized_inputs = normalized_inputs.reshape(batch.inputs.shape)
        else:
            normalized_inputs = batch.inputs

        return {
            self._placeholders[INPUTS]: normalized_inputs,
            self._placeholders[OUTPUT]: batch.output,
            self._placeholders[DROPOUT_KEEP_RATE]: self._hypers[DROPOUT_KEEP_RATE] if is_train else 1.0
        }

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
            self._placeholders[INPUTS] = tf.ones(shape=(1, ) + self._metadata[INPUT_SHAPE], dtype=tf.float32, name=INPUTS)
            self._placeholders[OUTPUT] = tf.ones(shape=(1), dtype=tf.int32, name=OUTPUT)
            self._placeholders[DROPOUT_KEEP_RATE] = 1.0

    def make_loss(self):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._ops[LOGITS_OP],
                                                                       labels=self._placeholders[OUTPUT])
        self._ops[LOSS_OP] = tf.reduce_mean(cross_entropy)
