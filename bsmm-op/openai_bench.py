from numpy.matrixlib.defmatrix import matrix
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

num_nonzero = int(matrix_size*matrix_size*sparsity)

minibatch_size = 64
rand = np.random.RandomState(seed=53)
with tf.device('GPU:0'):
    # Create a (random) sparsity pattern
    all_indices = np.arange(matrix_size)
    row_indices = np.sort(rand.choice(all_indices, size=num_nonzero, replace=True)).reshape(-1, 1)
    col_indices = rand.choice(all_indices, size=num_nonzero, replace=True).reshape(-1, 1)
    sparse_indices = np.concatenate([row_indices, col_indices], axis=-1)  # [NNZ, 2]


    # Initialize the sparse matrix multiplication object
    bsmm = BlocksparseMatMul(sparse_indices, block_size=block_size)

    # Input to graph
    x = tf.convert_to_tensor(rand.uniform(low=-2.0, high=2.0, size=(matrix_size, matrix_size*nIter)), dtype=tf.float32)

    # Initialize block-sparse weights
    w = rand.uniform(low=-2.0, high=2.0, size=num_nonzero)

    # Block-sparse matrix multiplication
    y = bsmm(x, w)

    # Run
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    start = time.perf_counter()
    result = sess.run([y], feed_dict = {})
    elapsed = time.perf_counter() - start
    print(elapsed)