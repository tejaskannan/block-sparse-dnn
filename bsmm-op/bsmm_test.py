
import tensorflow as tf
import numpy as np
import time
with tf.device('GPU:0'):
    #class BSMMTest(tf.test.TestCase):
        #def testBSMM(self):
    bsmm_module = tf.load_op_library("./bsmm.so")
    bsmm = bsmm_module.BCSRMatMulNA
    matrix_size = 2048
    block_size = 16
    blocks_per_row = 64
    rows = int(matrix_size/block_size)

    blocks = np.array([np.random.rand(block_size,block_size) for i in range(blocks_per_row*rows)])
    row_ptr = np.array([64*i for i in range((rows) + 1)], dtype=int)
    columns = np.array([],dtype=int)
    for i in range(rows):
        np.append(columns,np.random.choice(np.array(range(rows)), blocks_per_row))
    dense = np.random.rand(matrix_size, matrix_size)

    start = time.perf_counter_ns()
    output = bsmm(block_size=block_size, col_ids=columns, row_ptr=row_ptr, blocks=blocks, dense=dense)
    elapsed = time.perf_counter_ns() - start
    print(elapsed)