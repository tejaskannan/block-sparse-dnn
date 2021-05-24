from blocksparse.matmul import BlocksparseMatMul
import tensorflow as tf
import numpy as np
import time

hidden_size = 1024
block_size = 32
minibatch_size = 64

sparsity = 0.1
dims = int(hidden_size / block_size)
pattern = (np.random.uniform(low=0.0, high=1.0, size=(dims, dims)) < sparsity).astype(float) 

# Create a (random) sparsity pattern
#sparsity = np.random.randint(2, size=(hidden_size//block_size,hidden_size//block_size))
#sparsity = np.random.uniform(2, size=(hidden_size//block_size,hidden_size//block_size)i)

#Initialize the sparse matrix multiplication object
bsmm = BlocksparseMatMul(pattern, block_size=block_size)

# Input to graph
x = tf.placeholder(tf.float32, shape=[None, hidden_size])

# Initialize block-sparse weights
w = tf.get_variable("w", bsmm.w_shape, dtype=tf.float32)

# Block-sparse matrix multiplication
y = bsmm(x, w)

# Run
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

start = time.time()
for _ in range(1000):
    result = sess.run([y], feed_dict = {x: np.ones((hidden_size ,hidden_size), dtype='float32')})
end = time.time() 

print("Matmul took {} secs".format(end - start))
