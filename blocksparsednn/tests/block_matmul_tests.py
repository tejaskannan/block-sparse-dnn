import unittest
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from tensorflow.python.client import timeline

from blocksparsednn.layers.block_sparse_layer import BlockSparseLayer
from blocksparsednn.utils.tf_utils import block_diagonal_matmul


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

    def test_32_4(self):
        units = 32
        block_size = 4
        dims = int(units / block_size)

        rand = np.random.RandomState(seed=42)

        # Make a random pattern
        pattern = (rand.uniform(low=0.0, high=1.0, size=(dims, dims)) < 0.25).astype(float)  # [8, 8]

        # Make the input vector
        inputs = rand.uniform(low=-5.0, high=5.0, size=(1, units))  # [1, 32]

        # Execute the block sparse layer
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:

            input_ph = tf.compat.v1.placeholder(shape=[None, units], dtype=tf.float32)

            layer = BlockSparseLayer(pattern=pattern,
                                     block_size=block_size,
                                     units=units,
                                     initializer=tf.random_normal_initializer(),
                                     dtype=tf.float32)

            output = layer(inputs=input_ph)

            sess.run(tf.compat.v1.global_variables_initializer())
            result = sess.run(output, feed_dict={input_ph: inputs})

            trainable_vars = {var.name: var for var in sess.graph.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)}
            weights = sess.run(trainable_vars)

        # Make the corresponding dense matrix
        dense_mat = np.zeros(shape=(units, units))
        for row in range(dims):
            for col in range(dims):
                if np.isclose(pattern[row, col], 1.0):
                    row_st, row_end = row * block_size, (row + 1) * block_size
                    col_st, col_end = col * block_size, (col + 1) * block_size

                    block_value = weights['w-{0}-{1}:0'.format(row, col)]
                    dense_mat[row_st:row_end, col_st:col_end] = block_value

        expected = inputs.dot(dense_mat)  # [1, 32]
        
        self.assertTrue(np.all(np.isclose(expected, result)))

    def test_64_4(self):
        units = 64
        block_size = 4
        dims = int(units / block_size)

        rand = np.random.RandomState(seed=42)

        # Make a random pattern
        pattern = (rand.uniform(low=0.0, high=1.0, size=(dims, dims)) < 0.25).astype(float)  # [16, 16]

        # Make the input vector
        inputs = rand.uniform(low=-5.0, high=5.0, size=(1, units))  # [1, 64]

        # Execute the block sparse layer
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:

            input_ph = tf.compat.v1.placeholder(shape=[None, units], dtype=tf.float32)

            layer = BlockSparseLayer(pattern=pattern,
                                     block_size=block_size,
                                     units=units,
                                     initializer=tf.random_normal_initializer(),
                                     dtype=tf.float32)

            output = layer(inputs=input_ph)

            sess.run(tf.compat.v1.global_variables_initializer())
            result = sess.run(output, feed_dict={input_ph: inputs})

            trainable_vars = {var.name: var for var in sess.graph.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)}
            weights = sess.run(trainable_vars)

        # Make the corresponding dense matrix
        dense_mat = np.zeros(shape=(units, units))
        for row in range(dims):
            for col in range(dims):
                if np.isclose(pattern[row, col], 1.0):
                    row_st, row_end = row * block_size, (row + 1) * block_size
                    col_st, col_end = col * block_size, (col + 1) * block_size

                    block_value = weights['w-{0}-{1}:0'.format(row, col)]
                    dense_mat[row_st:row_end, col_st:col_end] = block_value

        expected = inputs.dot(dense_mat)  # [1, 64]
    
        # For some reason, the numerical values are slightly worse in TF
        self.assertTrue(np.all(np.abs(result - expected) < 1e-5))

    def test_128_16(self):
        units = 128
        block_size = 16
        dims = int(units / block_size)

        rand = np.random.RandomState(seed=42)

        # Make a random pattern
        pattern = (rand.uniform(low=0.0, high=1.0, size=(dims, dims)) < 0.25).astype(float)  # [8, 8]

        # Make the input vector
        inputs = rand.uniform(low=-5.0, high=5.0, size=(1, units))  # [1, 128]

        # Execute the block sparse layer
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:

            input_ph = tf.compat.v1.placeholder(shape=[None, units], dtype=tf.float32)

            layer = BlockSparseLayer(pattern=pattern,
                                     block_size=block_size,
                                     units=units,
                                     initializer=tf.random_normal_initializer(),
                                     dtype=tf.float32)

            output = layer(inputs=input_ph)

            sess.run(tf.compat.v1.global_variables_initializer())
            # result = sess.run(output, feed_dict={input_ph: inputs})

            run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
            run_metadata = tf.compat.v1.RunMetadata()
            result = sess.run(output, feed_dict={input_ph: inputs}, options=run_options, run_metadata=run_metadata)

            run_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = run_timeline.generate_chrome_trace_format()
            with open('run_meta_timeline.json', 'w') as fout:
                fout.write(chrome_trace)

            trainable_vars = {var.name: var for var in sess.graph.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)}
            weights = sess.run(trainable_vars)

        # Make the corresponding dense matrix
        dense_mat = np.zeros(shape=(units, units))
        for row in range(dims):
            for col in range(dims):
                if np.isclose(pattern[row, col], 1.0):
                    row_st, row_end = row * block_size, (row + 1) * block_size
                    col_st, col_end = col * block_size, (col + 1) * block_size

                    block_value = weights['w-{0}-{1}:0'.format(row, col)]
                    dense_mat[row_st:row_end, col_st:col_end] = block_value

        expected = inputs.dot(dense_mat)  # [1, 128]
    
        # For some reason, the numerical values are slightly worse in TF
        self.assertTrue(np.all(np.abs(result - expected) < 1e-5))


class TestBlockDiagonal(unittest.TestCase):

    def test_even(self):
        batch_size = 32
        feature_size = 16
        block_size = 4

        rand = np.random.RandomState(seed=43)

        dense_mat = rand.uniform(size=(batch_size, feature_size))
        blocks = [rand.uniform(size=(block_size, block_size)) for _ in range(4)]

        block_dense = np.zeros(shape=(feature_size, feature_size))
        block_dense[0:4, 0:4] = blocks[0]
        block_dense[4:8, 4:8] = blocks[1]
        block_dense[8:12, 8:12] = blocks[2]
        block_dense[12:16, 12:16] = blocks[3]
    
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            dense_ph = tf.compat.v1.placeholder(shape=[batch_size, feature_size],
                                                dtype=tf.float32,
                                                name='dense-ph')

            feed_dict = {
                dense_ph: dense_mat
            }

            block_phs: List[tf.compat.v1.placeholder] = []
            for i, block in enumerate(blocks):
                block_ph = tf.compat.v1.placeholder(shape=[block_size, block_size],
                                                    dtype=tf.float32,
                                                    name='block-ph-{0}'.format(i))
                block_phs.append(block_ph)

                feed_dict[block_ph] = block

            output = block_diagonal_matmul(dense_ph, block_phs)
            
            result = sess.run(output, feed_dict=feed_dict)
            
        expected = np.matmul(dense_mat, block_dense)

        self.assertTrue(np.all(np.isclose(result, expected)))

    def test_expand(self):
        batch_size = 32
        feature_size = 16
        in_block_size = 4
        out_block_size = 8

        rand = np.random.RandomState(seed=32)

        dense_mat = rand.uniform(size=(batch_size, feature_size))
        blocks = [rand.uniform(size=(in_block_size, out_block_size)) for _ in range(4)]

        block_dense = np.zeros(shape=(16, 32))
        block_dense[0:4, 0:8] = blocks[0]
        block_dense[4:8, 8:16] = blocks[1]
        block_dense[8:12, 16:24] = blocks[2]
        block_dense[12:16, 24:32] = blocks[3]
    
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            dense_ph = tf.compat.v1.placeholder(shape=[batch_size, feature_size],
                                                dtype=tf.float32,
                                                name='dense-ph')

            feed_dict = {
                dense_ph: dense_mat
            }

            block_phs: List[tf.compat.v1.placeholder] = []
            for i, block in enumerate(blocks):
                block_ph = tf.compat.v1.placeholder(shape=[in_block_size, out_block_size],
                                                    dtype=tf.float32,
                                                    name='block-ph-{0}'.format(i))
                block_phs.append(block_ph)

                feed_dict[block_ph] = block

            output = block_diagonal_matmul(dense_ph, block_phs)
            
            result = sess.run(output, feed_dict=feed_dict)
            
        expected = np.matmul(dense_mat, block_dense)

        self.assertTrue(np.all(np.isclose(result, expected)))

    def test_contract(self):
        batch_size = 32
        feature_size = 16
        in_block_size = 4
        out_block_size = 2

        rand = np.random.RandomState(seed=32)

        dense_mat = rand.uniform(size=(batch_size, feature_size))
        blocks = [rand.uniform(size=(in_block_size, out_block_size)) for _ in range(4)]

        block_dense = np.zeros(shape=(16, 8))
        block_dense[0:4, 0:2] = blocks[0]
        block_dense[4:8, 2:4] = blocks[1]
        block_dense[8:12, 4:6] = blocks[2]
        block_dense[12:16, 6:8] = blocks[3]
    
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            dense_ph = tf.compat.v1.placeholder(shape=[batch_size, feature_size],
                                                dtype=tf.float32,
                                                name='dense-ph')

            feed_dict = {
                dense_ph: dense_mat
            }

            block_phs: List[tf.compat.v1.placeholder] = []
            for i, block in enumerate(blocks):
                block_ph = tf.compat.v1.placeholder(shape=[in_block_size, out_block_size],
                                                    dtype=tf.float32,
                                                    name='block-ph-{0}'.format(i))
                block_phs.append(block_ph)

                feed_dict[block_ph] = block

            output = block_diagonal_matmul(dense_ph, block_phs)
            
            result = sess.run(output, feed_dict=feed_dict)
            
        expected = np.matmul(dense_mat, block_dense)

        self.assertTrue(np.all(np.isclose(result, expected)))


if __name__ == '__main__':
    unittest.main()
