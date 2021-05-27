import unittest
import tensorflow as tf
import numpy as np
import scipy.sparse as sp


class SparseEmbeddingLookup(unittest.TestCase):

    def test_embedding_lookup(self):
        rand = np.random.RandomState(seed=126)
        dim0, dim1 = 3, 4

        # Make a small random sparse matrix
        mask = (rand.uniform(low=0.0, high=1.0, size=(dim0, dim1)) < 0.5).astype(float)
        dense_weights = rand.uniform(low=-5.0, high=5.0, size=mask.shape) * mask
        
        sp_weights = sp.coo_matrix(dense_weights)
        sp_indices = [(row, col) for row, col in zip(sp_weights.row, sp_weights.col)]

        dense_mat = rand.uniform(low=-5.0, high=5.0, size=mask.shape).T
        # dense_mat = np.ones_like(mask).T
        expected = sp_weights.dot(dense_mat)

        # Execute in TF with embedding lookup
        with tf.Session(graph=tf.Graph()) as sess:
            
            dense_input = tf.placeholder(shape=dense_mat.shape,
                                                   dtype=tf.float32,
                                                   name='dense-input')

            sp_ids = tf.sparse.SparseTensor(indices=sp_indices,
                                            values=sp_weights.col,
                                            dense_shape=dense_weights.shape)

            sp_data = tf.sparse.SparseTensor(indices=sp_indices,
                                             values=sp_weights.data,
                                             dense_shape=dense_weights.shape)

            output = tf.nn.embedding_lookup_sparse(params=dense_input,
                                                   sp_ids=sp_ids,
                                                   sp_weights=sp_data,
                                                   combiner='sum')

            result = sess.run(output, feed_dict={dense_input: dense_mat})

        self.assertTrue(np.all(np.abs(result - expected) < 1e-5))


if __name__ == '__main__':
    unittest.main()

