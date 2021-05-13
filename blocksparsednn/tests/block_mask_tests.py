import unittest
import tensorflow as tf
import numpy as np

from blocksparsednn.utils.tf_utils import project_block_mask


class BlockProjectionTests(unittest.TestCase):

    def test_project_two_by_two(self):
        with tf.Session() as sess:
            block_mask = tf.constant([[1.0, 0.0], [0.0, 1.0]])
            projected = project_block_mask(block_mask, block_size=2)
            predicted = sess.run(projected)

        expected = np.zeros(shape=(4, 4))
        expected[0:2, 0:2] = 1.0
        expected[2:4, 2:4] = 1.0

        self.assertTrue(np.all(np.isclose(expected, predicted)))

    def test_project_two_by_four(self):
        with tf.Session() as sess:
            block_mask = tf.constant([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
            projected = project_block_mask(block_mask, block_size=2)
            predicted = sess.run(projected)

        expected = np.zeros(shape=(4, 8))
        expected[0:2, 0:2] = 1.0
        expected[0:2, 6:8] = 1.0
        expected[2:4, 2:4] = 1.0
        expected[2:4, 4:6] = 1.0

        self.assertTrue(np.all(np.isclose(expected, predicted)))


if __name__ == '__main__':
    unittest.main()

