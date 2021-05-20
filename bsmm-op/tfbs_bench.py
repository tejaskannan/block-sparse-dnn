import tensorflow as tf
import numpy as np
import time
import sys

matrix_size = int(sys.argv[1])
block_size = int(sys.argv[2])
rows = int(matrix_size/block_size)
nIter = int(sys.argv[4])
sparsity = float(sys.argv[3])

with tf.device('GPU:0'):
    DIMS = matrix_size
    SPARSITY = sparsity

    num_nonzero = int(DIMS * DIMS * SPARSITY)

    # Generate the sparse matrix
    rand = np.random.RandomState(seed=53)

    all_indices = np.arange(DIMS)
    row_indices = np.sort(rand.choice(all_indices, size=num_nonzero, replace=True)).reshape(-1, 1)
    col_indices = rand.choice(all_indices, size=num_nonzero, replace=True).reshape(-1, 1)
    sparse_indices = np.concatenate([row_indices, col_indices], axis=-1)  # [NNZ, 2]

    sparse_values = rand.uniform(low=-2.0, high=2.0, size=num_nonzero) # [NNZ]

    
    graph = tf.Graph()

    with graph.as_default():
        inputs = tf.convert_to_tensor(rand.uniform(low=-2.0, high=2.0, size=(DIMS, DIMS*nIter)), dtype=tf.float32)

        weights = tf.compat.v1.placeholder(shape=[num_nonzero],
                                        dtype=tf.float32,
                                        name='sp-weights')

        indices = tf.compat.v1.placeholder(shape=[num_nonzero, 2],
                                        dtype=tf.int64,
                                        name='sp-indices')


        sp_mat = tf.sparse.SparseTensor(values=weights, indices=indices, dense_shape=[DIMS, DIMS])

        output = tf.sparse.sparse_dense_matmul(sp_mat, inputs)


    with tf.compat.v1.Session(graph=graph) as sess:
        tf.compat.v1.initialize_all_variables().run()
        
        

        
        
        feed_dict = {
            weights: sparse_values,
            indices: sparse_indices
        }
        start = time.perf_counter()
        result = sess.run(output, feed_dict=feed_dict)
        elapsed = time.perf_counter() - start
        print(f"BS Mat Mul in: {elapsed}")
