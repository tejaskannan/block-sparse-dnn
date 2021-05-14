import tensorflow as tf
import numpy as np
import time

DIMS = 1024
SPARSITY = 0.1
TRIALS = 1000

# Generate the sparse matrix
rand = np.random.RandomState(seed=53)

pattern = (rand.uniform(low=0.0, high=1.0, size=(DIMS, DIMS)) < SPARSITY).astype(float)
dense_mat = rand.uniform(low=-2.0, high=2.0, size=(DIMS, DIMS)) * pattern

# Generate the input matrix
input_mat = rand.uniform(low=-2.0, high=2.0, size=(DIMS, DIMS))

with tf.Session(graph=tf.Graph()) as sess:
    
    inputs = tf.placeholder(shape=[DIMS, DIMS],
                            dtype=tf.float32,
                            name='inputs')

    weights = tf.get_variable(shape=[DIMS, DIMS],
                              dtype=tf.float32,
                              initializer=tf.glorot_uniform_initializer(),
                              name='weights')

    output = tf.matmul(inputs, weights)

    sess.run(tf.compat.v1.global_variables_initializer())

    feed_dict = {
        inputs: dense_mat
    }

    start = time.time()
    for _ in range(TRIALS):
        result = sess.run(output, feed_dict=feed_dict)

    end = time.time()

    print('Sparse Matmul took {0} second.'.format(end - start))
