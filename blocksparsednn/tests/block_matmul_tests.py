import unittest
import tensorflow as tf
import numpy as np

from blocksparsednn.layers.block_sparse_layer import BlockSparseLayer


class TestBlockSparseMatMul(unittest.TestCase):

    def test_ones_single(self):
        pattern = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 0]])
        block_size = 1
        units = 3

        inputs = np.array([[1, 2, 3]])
        expected = np.matmul(inputs, pattern)

        with tf.compat.v1.Session(graph=tf.Graph()) as sess:

            input_ph = tf.compat.v1.placeholder(shape=[None, units], dtype=tf.float32)

            layer = BlockSparseLayer(pattern=pattern,
                                     block_size=block_size,
                                     units=units,
                                     initializer=tf.compat.v1.initializers.ones(),
                                     dtype=tf.float32)

            output = layer(inputs=input_ph)

            sess.run(tf.compat.v1.global_variables_initializer())
            result = sess.run(output, feed_dict={input_ph: inputs})

        self.assertTrue(np.all(np.isclose(result, expected)))

    def test_ones_multiple(self):
        pattern = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 0]])
        block_size = 2
        units = 6

        inputs = np.array([[1, 2, 3, 4, 5, 6]])

        dense_mat = np.zeros(shape=(units, units))
        dense_mat[0:2, 0:2] = 1
        dense_mat[0:2, 4:6] = 1
        dense_mat[2:4, 2:4] = 1
        dense_mat[2:4, 4:6] = 1
        dense_mat[4:6, 0:2] = 1

        expected = np.matmul(inputs, dense_mat)

        with tf.compat.v1.Session(graph=tf.Graph()) as sess:

            input_ph = tf.compat.v1.placeholder(shape=[None, units], dtype=tf.float32)

            layer = BlockSparseLayer(pattern=pattern,
                                     block_size=block_size,
                                     units=units,
                                     initializer=tf.compat.v1.initializers.ones(),
                                     dtype=tf.float32)

            output = layer(inputs=input_ph)

            sess.run(tf.compat.v1.global_variables_initializer())
            result = sess.run(output, feed_dict={input_ph: inputs})

        self.assertTrue(np.all(np.isclose(result, expected)))

    def test_constant_multiple(self):
        pattern = np.array([[0, 1, 0], [1, 1, 0], [0, 1, 1]])
        block_size = 2
        units = 6
        k = 3

        inputs = np.array([[2, 4, 6, 8, 10, 12]])

        dense_mat = np.zeros(shape=(units, units))
        dense_mat[0:2, 2:4] = k
        dense_mat[2:4, 0:2] = k
        dense_mat[2:4, 2:4] = k
        dense_mat[4:6, 2:4] = k
        dense_mat[4:6, 4:6] = k

        expected = np.matmul(inputs, dense_mat)

        with tf.compat.v1.Session(graph=tf.Graph()) as sess:

            input_ph = tf.compat.v1.placeholder(shape=[None, units], dtype=tf.float32)

            layer = BlockSparseLayer(pattern=pattern,
                                     block_size=block_size,
                                     units=units,
                                     initializer=tf.compat.v1.initializers.constant(k),
                                     dtype=tf.float32)

            output = layer(inputs=input_ph)

            sess.run(tf.compat.v1.global_variables_initializer())
            result = sess.run(output, feed_dict={input_ph: inputs})

        self.assertTrue(np.all(np.isclose(result, expected)))


if __name__ == '__main__':
    unittest.main()
