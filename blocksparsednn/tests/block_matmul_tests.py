import unittest
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from tensorflow.python.client import timeline

from blocksparsednn.layers.block_sparse_layer import BlockSparseLayer
from blocksparsednn.utils.tf_utils import block_diagonal_matmul, block_sparse_matmul


class TestBlockSparseMatMul(unittest.TestCase):

    def test_ones_single(self):
        pattern = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 0]])
        block_size = 1
        units = 3
        num_blocks = np.sum(pattern)

        inputs = np.array([[1, 2, 3]])
        expected = np.matmul(inputs, pattern)

        with tf.Session(graph=tf.Graph()) as sess:

            input_ph = tf.placeholder(shape=[None, units], dtype=tf.float32)
            feed_dict = {
                input_ph: inputs
            }
            

            blocks: List[tf.placeholder] = [] 
            for i in range(num_blocks):
                block_ph = tf.placeholder(shape=[block_size, block_size],
                                                    dtype=tf.float32,
                                                    name='block-{0}'.format(i))
                block = np.ones(shape=(block_size, block_size))

                feed_dict[block_ph] = block
                blocks.append(block_ph)
            
            rows_ph = tf.placeholder(shape=[num_blocks],
                                               dtype=tf.int32,
                                               name='rows')

            cols_ph = tf.placeholder(shape=[num_blocks],
                                               dtype=tf.int32,
                                               name='cols')

            feed_dict[rows_ph] = [0, 0, 1, 1, 2]
            feed_dict[cols_ph] = [0, 2, 1, 2, 0]

            output = block_sparse_matmul(dense_mat=input_ph,
                                         blocks=blocks,
                                         nonzero_rows=rows_ph,
                                         nonzero_cols=cols_ph,
                                         output_dims=units)

            sess.run(tf.global_variables_initializer())
            result = sess.run(output, feed_dict=feed_dict)

        self.assertTrue(np.all(np.isclose(result, expected)))

    def test_ones_multiple(self):
        pattern = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 0]])
        block_size = 2
        units = 6
        num_blocks = np.sum(pattern)

        inputs = np.array([[1, 2, 3, 4, 5, 6]])

        dense_mat = np.zeros(shape=(units, units))
        dense_mat[0:2, 0:2] = 1
        dense_mat[0:2, 4:6] = 1
        dense_mat[2:4, 2:4] = 1
        dense_mat[2:4, 4:6] = 1
        dense_mat[4:6, 0:2] = 1

        expected = np.matmul(inputs, dense_mat)

        with tf.Session(graph=tf.Graph()) as sess:

            input_ph = tf.placeholder(shape=[None, units], dtype=tf.float32)
            feed_dict = {
                input_ph: inputs
            }

            blocks: List[tf.placeholder] = [] 
            for i in range(num_blocks):
                block_ph = tf.placeholder(shape=[block_size, block_size],
                                                    dtype=tf.float32,
                                                    name='block-{0}'.format(i))
                block = np.ones(shape=(block_size, block_size))

                feed_dict[block_ph] = block
                blocks.append(block_ph)
            
            rows_ph = tf.placeholder(shape=[num_blocks],
                                               dtype=tf.int32,
                                               name='rows')

            cols_ph = tf.placeholder(shape=[num_blocks],
                                               dtype=tf.int32,
                                               name='cols')

            feed_dict[rows_ph] = [0, 0, 1, 1, 2]
            feed_dict[cols_ph] = [0, 2, 1, 2, 0]

            output = block_sparse_matmul(dense_mat=input_ph,
                                         blocks=blocks,
                                         nonzero_rows=rows_ph,
                                         nonzero_cols=cols_ph,
                                         output_dims=units)

            sess.run(tf.global_variables_initializer())
            result = sess.run(output, feed_dict=feed_dict)

        self.assertTrue(np.all(np.isclose(result, expected)))

    def test_multiple_varying(self):
        pattern = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 0]])
        block_size = 2
        units = 6
        num_blocks = np.sum(pattern)

        inputs = np.array([[1, 2, 3, 4, 5, 6]])

        rand = np.random.RandomState(seed=51)
        blocks = [rand.uniform(low=-1.0, high=1.0, size=(block_size, block_size)) for _ in range(num_blocks)]

        dense_mat = np.zeros(shape=(units, units))
        dense_mat[0:2, 0:2] = blocks[0]
        dense_mat[0:2, 4:6] = blocks[1]
        dense_mat[2:4, 2:4] = blocks[2]
        dense_mat[2:4, 4:6] = blocks[3]
        dense_mat[4:6, 0:2] = blocks[4]

        expected = np.matmul(inputs, dense_mat)

        with tf.Session(graph=tf.Graph()) as sess:

            input_ph = tf.placeholder(shape=[None, units], dtype=tf.float32)
            feed_dict = {
                input_ph: inputs
            }

            blocks_list: List[tf.placeholder] = [] 
            for i in range(num_blocks):
                block_ph = tf.placeholder(shape=[block_size, block_size],
                                                    dtype=tf.float32,
                                                    name='block-{0}'.format(i))
                block = blocks[i]

                feed_dict[block_ph] = block
                blocks_list.append(block_ph)
            
            rows_ph = tf.placeholder(shape=[num_blocks],
                                               dtype=tf.int32,
                                               name='rows')

            cols_ph = tf.placeholder(shape=[num_blocks],
                                               dtype=tf.int32,
                                               name='cols')

            feed_dict[rows_ph] = [0, 0, 1, 1, 2]
            feed_dict[cols_ph] = [0, 2, 1, 2, 0]

            output = block_sparse_matmul(dense_mat=input_ph,
                                         blocks=blocks_list,
                                         nonzero_rows=rows_ph,
                                         nonzero_cols=cols_ph,
                                         output_dims=units)

            sess.run(tf.global_variables_initializer())
            result = sess.run(output, feed_dict=feed_dict)

        self.assertTrue(np.all(np.isclose(result, expected)))

    def test_32_4(self):
        units = 32
        block_size = 4
        dims = int(units / block_size)

        rand = np.random.RandomState(seed=42)

        # Make a random pattern
        pattern = (rand.uniform(low=0.0, high=1.0, size=(dims, dims)) < 0.25).astype(float)  # [8, 8]
        num_blocks = int(np.sum(pattern))

        # Make the corresponding dense matrix
        dense_mat = np.zeros(shape=(units, units))
        
        blocks: List[np.ndarray] = []
        rows: List[int] = []
        cols: List[int] = []

        for row in range(dims):
            for col in range(dims):
                if np.isclose(pattern[row, col], 1.0):
                    row_st, row_end = row * block_size, (row + 1) * block_size
                    col_st, col_end = col * block_size, (col + 1) * block_size

                    block_value = rand.uniform(low=-1.0, high=1.0, size=(block_size, block_size))
                    dense_mat[row_st:row_end, col_st:col_end] = block_value

                    blocks.append(block_value)
                    rows.append(row)
                    cols.append(col)

        # Make the input vector
        inputs = rand.uniform(low=-5.0, high=5.0, size=(2, units))  # [2, 32]

        expected = inputs.dot(dense_mat)  # [2, 32]

        with tf.Session(graph=tf.Graph()) as sess:

            input_ph = tf.placeholder(shape=[None, units], dtype=tf.float32)
            feed_dict = {
                input_ph: inputs
            }

            blocks_list: List[tf.placeholder] = [] 
            for i in range(num_blocks):
                block_ph = tf.placeholder(shape=[block_size, block_size],
                                                    dtype=tf.float32,
                                                    name='block-{0}'.format(i))
                block = blocks[i]

                feed_dict[block_ph] = block
                blocks_list.append(block_ph)
            
            rows_ph = tf.placeholder(shape=[num_blocks],
                                               dtype=tf.int32,
                                               name='rows')

            cols_ph = tf.placeholder(shape=[num_blocks],
                                               dtype=tf.int32,
                                               name='cols')

            feed_dict[rows_ph] = rows
            feed_dict[cols_ph] = cols

            output = block_sparse_matmul(dense_mat=input_ph,
                                         blocks=blocks_list,
                                         nonzero_rows=rows_ph,
                                         nonzero_cols=cols_ph,
                                         output_dims=units)

            sess.run(tf.global_variables_initializer())

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            result = sess.run(output, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)

            run_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = run_timeline.generate_chrome_trace_format()
            with open('run_timeline.json', 'w') as fout:
                fout.write(chrome_trace)

            # result = sess.run(output, feed_dict=feed_dict)

        self.assertTrue(np.all(np.abs(expected - result) < 1e-5))

    def test_expand(self):
        pattern = np.array([[1, 0, 1, 1], [0, 1, 1, 0], [1, 0, 0, 1]])
        block_size = 2
        in_units = 6
        out_units = 8
        num_blocks = np.sum(pattern)

        inputs = np.array([[1, 2, 3, 4, 5, 6]])

        rand = np.random.RandomState(seed=51)
        blocks = [rand.uniform(low=-1.0, high=1.0, size=(block_size, block_size)) for _ in range(num_blocks)]

        dense_mat = np.zeros(shape=(in_units, out_units))
        dense_mat[0:2, 0:2] = blocks[0]
        dense_mat[0:2, 4:6] = blocks[1]
        dense_mat[0:2, 6:8] = blocks[2]
        dense_mat[2:4, 2:4] = blocks[3]
        dense_mat[2:4, 4:6] = blocks[4]
        dense_mat[4:6, 0:2] = blocks[5]
        dense_mat[4:6, 6:8] = blocks[6]

        expected = np.matmul(inputs, dense_mat)

        with tf.Session(graph=tf.Graph()) as sess:

            input_ph = tf.placeholder(shape=[None, in_units], dtype=tf.float32)
            feed_dict = {
                input_ph: inputs
            }

            blocks_list: List[tf.placeholder] = [] 
            for i in range(num_blocks):
                block_ph = tf.placeholder(shape=[block_size, block_size],
                                                    dtype=tf.float32,
                                                    name='block-{0}'.format(i))
                block = blocks[i]

                feed_dict[block_ph] = block
                blocks_list.append(block_ph)
            
            rows_ph = tf.placeholder(shape=[num_blocks],
                                               dtype=tf.int32,
                                               name='rows')

            cols_ph = tf.placeholder(shape=[num_blocks],
                                               dtype=tf.int32,
                                               name='cols')

            feed_dict[rows_ph] = [0, 0, 0, 1, 1, 2, 2]
            feed_dict[cols_ph] = [0, 2, 3, 1, 2, 0, 3]

            output = block_sparse_matmul(dense_mat=input_ph,
                                         blocks=blocks_list,
                                         nonzero_rows=rows_ph,
                                         nonzero_cols=cols_ph,
                                         output_dims=out_units)

            sess.run(tf.global_variables_initializer())
            result = sess.run(output, feed_dict=feed_dict)

        self.assertTrue(np.all(np.isclose(result, expected)))

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
    
        with tf.Session(graph=tf.Graph()) as sess:
            dense_ph = tf.placeholder(shape=[batch_size, feature_size],
                                                dtype=tf.float32,
                                                name='dense-ph')

            feed_dict = {
                dense_ph: dense_mat
            }

            block_phs: List[tf.placeholder] = []
            for i, block in enumerate(blocks):
                block_ph = tf.placeholder(shape=[block_size, block_size],
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
    
        with tf.Session(graph=tf.Graph()) as sess:
            dense_ph = tf.placeholder(shape=[batch_size, feature_size],
                                                dtype=tf.float32,
                                                name='dense-ph')

            feed_dict = {
                dense_ph: dense_mat
            }

            block_phs: List[tf.placeholder] = []
            for i, block in enumerate(blocks):
                block_ph = tf.placeholder(shape=[in_block_size, out_block_size],
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
    
        with tf.Session(graph=tf.Graph()) as sess:
            dense_ph = tf.placeholder(shape=[batch_size, feature_size],
                                                dtype=tf.float32,
                                                name='dense-ph')

            feed_dict = {
                dense_ph: dense_mat
            }

            block_phs: List[tf.placeholder] = []
            for i, block in enumerate(blocks):
                block_ph = tf.placeholder(shape=[in_block_size, out_block_size],
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
