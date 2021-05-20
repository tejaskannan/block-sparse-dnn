from blocksparse.matmul import BlocksparseMatMul
import tensorflow as tf
import numpy as np
import time
import sys

matrix_size = int(sys.argv[1])
block_size = int(sys.argv[2])
rows = int(matrix_size/block_size)
nIter = int(sys.argv[4])
sparsity = float(sys.argv[3])

minibatch_size = 64
rand = np.random.RandomState(seed=53)
with tf.device('GPU:0'):
    # Create a (random) sparsity pattern
    sparsity = np.random.randint(2, size=(matrix_size//block_size,matrix_size//block_size))

    # Initialize the sparse matrix multiplication object
    bsmm = BlocksparseMatMul(sparsity, block_size=block_size)

    # Input to graph
    x = tf.convert_to_tensor(rand.uniform(low=-2.0, high=2.0, size=(matrix_size, matrix_size*nIter)), dtype=tf.float32)

    # Initialize block-sparse weights
    w = tf.get_variable("w", bsmm.w_shape, dtype=tf.float32)

    # Block-sparse matrix multiplication
    y = bsmm(x, w)

    # Run
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    start = time.perf_counter()
    result = sess.run([y], feed_dict = {})
    elapsed = time.perf_counter() - start
    print(elapsed)