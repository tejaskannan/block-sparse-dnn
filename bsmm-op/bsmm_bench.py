import tensorflow as tf
import numpy as np
import time
import sys
import math

matrix_size = int(sys.argv[1])
block_size = int(sys.argv[2])
rows = int(matrix_size/block_size)
nIter = int(sys.argv[4])
sparsity = float(sys.argv[3])
blocks_per_row = int(math.ceil((matrix_size * matrix_size * sparsity)/(block_size*block_size*rows)))
rand = np.random.RandomState(seed=53)

with tf.device('GPU:0'):
    #class BSMMTest(tf.test.TestCase):
        #def testBSMM(self):
    bsmm_module = tf.load_op_library("./bsmm.so")
    bsmm = bsmm_module.BCSRMatMul
    

    blocks_in = np.array([np.random.rand(block_size,block_size) for j in range(blocks_per_row*rows)])
    row_ptr_in = np.array([64*j for j in range((rows) + 1)], dtype=np.uint64)
    cols = np.array([],dtype=np.uint64)
    for j in range(rows):
        cols= np.append(cols,np.random.choice(np.array(range(rows)), blocks_per_row))
    graph = tf.Graph()

    with graph.as_default():
        inputs = tf.convert_to_tensor(rand.uniform(low=-2.0, high=2.0, size=(matrix_size, matrix_size*nIter)), dtype=tf.float32)

        columns = tf.compat.v1.placeholder(shape=[blocks_per_row*rows],
                                        dtype=tf.uint64,
                                        name='sp-cols')

        blocks = tf.compat.v1.placeholder(shape=[blocks_per_row*rows, block_size, block_size],
                                        dtype=tf.float32,
                                        name='sp-blocks')

        row_ptr = tf.compat.v1.placeholder(shape=[rows+1],
                                        dtype=tf.uint64,
                                        name='sp-row_ptr')


        output = bsmm(block_size=block_size, col_ids=columns, row_ptr=row_ptr, blocks=blocks, dense=inputs)

    with tf.compat.v1.Session(graph=graph) as sess:
        tf.compat.v1.initialize_all_variables().run()
        
        
        feed_dict = {
                columns: cols,
                blocks: blocks_in,
                row_ptr : row_ptr_in
        }

        start = time.perf_counter()
        sess.run(output, feed_dict=feed_dict)
        elapsed = time.perf_counter() - start

        print(elapsed)


