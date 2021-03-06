from blocksparse.matmul import BlocksparseMatMul
import tensorflow as tf
import numpy as np
import time
from typing import List

#NOTE: CHANGE THESE PARAMETERS TO CHANGE THE EXPERIMENT
hidden_size = 256
block_size = 32
sparsity = 0.05
trials = 100

dims = int(hidden_size / block_size)

# Create a (random) sparsity pattern
rand = np.random.RandomState(seed=51)
indices = np.arange(dims * dims)
block_indices = rand.choice(indices, size=int(round(sparsity * dims * dims)), replace=False)

pattern = np.zeros(shape=(dims, dims))
for idx in block_indices:
    row = int(idx / dims)
    col = int(idx % dims)

    pattern[row, col] = 1

print('Detected Sparsity: {0:.4f}'.format(np.sum(pattern) / (dims * dims)))

shuffle_indices = np.arange(dims)
rand.shuffle(np.arange(dims))

times: List[float] = []

with tf.Session(graph=tf.Graph()) as sess:
 
    bsmm = BlocksparseMatMul(pattern, block_size=block_size)
   
    inputs = tf.placeholder(shape=(hidden_size, hidden_size), dtype=tf.float32, name='inputs')
    weights = tf.get_variable(name='W', shape=bsmm.w_shape, dtype=tf.float32)
    shuffle_weights = tf.get_variable(name='shuffle', shape=(1, hidden_size), dtype=tf.float32)

    transformed = bsmm(inputs, weights)

    shuffled_values = tf.gather(transformed, shuffle_indices, axis=-1, name='shuffle-gather')
    shuffled_transformed = tf.multiply(shuffled_values, shuffle_weights)

    transformed = transformed + shuffled_transformed

    sess.run(tf.global_variables_initializer())

    for i in range(trials + 1):
        mat = rand.uniform(low=-2.0, high=2.0, size=(hidden_size, hidden_size))

        start = time.perf_counter()
        sess.run(transformed, feed_dict={inputs: mat})
        end = time.perf_counter()

        elapsed = end - start
        times.append(elapsed)

avg_time = np.average(times)
avg_throughput = 1.0 / avg_time
print('Time to perform {0} x {0} MatMul @ {1:.3f} Sparsity With Shuffling: {2:.6f} secs / op, {3:.6f} ops / sec'.format(dims, sparsity, avg_time, avg_throughput))
