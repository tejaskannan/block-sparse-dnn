from blocksparse.matmul import BlocksparseMatMul
import tensorflow as tf
import numpy as np
import datetime

hidden_size = 4096
block_size = 32
minibatch_size = 64

# Create a (random) sparsity pattern
sparsity = np.random.randint(2, size=(hidden_size//block_size,hidden_size//block_size))

# Run
with tf.Session(graph=tf.Graph()) as sess:
    # Initialize the sparse matrix multiplication object
    bsmm = BlocksparseMatMul(sparsity, block_size=block_size)

    # Input to graph
    x = tf.placeholder(tf.float32, shape=[None, hidden_size])

    # Initialize block-sparse weights
    w = tf.get_variable("w", bsmm.w_shape, dtype=tf.float32)

    # Block-sparse matrix multiplication
    y = bsmm(x, w)

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    grads = tf.gradients(y, [w])
    

    start = datetime.time()
    for i in range(1000):
        result = sess.run([grads], feed_dict = {x: np.ones((minibatch_size,hidden_size), dtype='float32')})
    finish = datetime.time() 

    print("Matmul took {}".format(finish-start))
