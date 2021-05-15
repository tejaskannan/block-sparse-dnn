import tensorflow as tf
import numpy as np
import time
import sys

matrix_size = int(sys.argv[1])
block_size = int(sys.argv[2])
blocks_per_row = int(sys.argv[3])
rows = int(matrix_size/block_size)
nIter = int(sys.argv[4])
sparsity = (block_size*block_size*blocks_per_row*rows)/(matrix_size*matrix_size)

with tf.device('GPU:0'):
    #class BSMMTest(tf.test.TestCase):
        #def testBSMM(self):
    bsmm_module = tf.load_op_library("./bsmm.so")
    bsmm = bsmm_module.BCSRMatMul
    
    blocks = []
    row_ptr = []
    columns = []
    dense = []

    for i in range(nIter):
        blocks_tmp= np.array([np.random.rand(block_size,block_size) for j in range(blocks_per_row*rows)])
        row_ptr_tmp = np.array([64*j for j in range((rows) + 1)], dtype=int)
        columns_tmp = np.array([],dtype=int)
        for j in range(rows):
            np.append(columns_tmp,np.random.choice(np.array(range(rows)), blocks_per_row))
        dense_tmp = np.random.rand(matrix_size, matrix_size)

        blocks.append(blocks_tmp)
        row_ptr.append(row_ptr_tmp)
        columns.append(columns_tmp)
        dense.append(dense_tmp)
    
    blocks = tf.convert_to_tensor(np.array(blocks))
    columns = tf.convert_to_tensor(np.array(columns))
    row_ptr = tf.convert_to_tensor(np.array(row_ptr))
    dense = tf.convert_to_tensor(np.array(dense))
    
    #print("Initialization complete")
    start = time.perf_counter()
    for i in range(nIter):
        output = bsmm(block_size=block_size, col_ids=columns[i], row_ptr=row_ptr[i], blocks=blocks[i], dense=dense[i])
    elapsed = time.perf_counter() - start
    #print(f"{nIter} BCSRMatMul in: {elapsed}")
    print(elapsed)


