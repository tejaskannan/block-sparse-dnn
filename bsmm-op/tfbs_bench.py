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
    num_nonzero = int(matrix_size * matrix_size * sparsity)

    # Generate the sparse matrix
    rand = np.random.RandomState(seed=53)

    all_indices = []
    row_indices = []
    col_indices = []
    sparse_indices = []
    sparse_values = []
    input_mat = []

    for i in range(nIter):
        all_indices.append(np.arange(matrix_size))
        row_indices.append(np.sort(rand.choice(all_indices[i], size=num_nonzero, replace=True)).reshape(-1, 1))
        col_indices.append(rand.choice(all_indices[i], size=num_nonzero, replace=True).reshape(-1, 1))
        sparse_indices.append(np.concatenate([row_indices[i], col_indices[i]], axis=-1))  # [NNZ, 2]

        sparse_values.append(rand.uniform(low=-2.0, high=2.0, size=num_nonzero)) # [NNZ]

        # Generate the input matrix
        input_mat.append(rand.uniform(low=-2.0, high=2.0, size=(matrix_size, matrix_size)))

    # all_indices = tf.convert_to_tensor(np.array(all_indices))
    # row_indices = tf.convert_to_tensor(np.array(row_indices))
    # col_indices = tf.convert_to_tensor(np.array(col_indices))
    # sparse_indices = tf.convert_to_tensor(np.array(sparse_indices))
    # sparse_values = tf.convert_to_tensor(np.array(sparse_values))
    input_mat = tf.convert_to_tensor(np.array(input_mat))

    sp_mat = []
    for i in range(nIter):
        sp_mat.append(tf.sparse.SparseTensor(values=sparse_values[i], indices=sparse_indices[i], dense_shape=[matrix_size,matrix_size]))

    total_pattern = tf.sparse.concat(axis=1, sp_inputs=sp_mat)

    #print("Initialization complete")
    start = time.perf_counter()
    for i in range(nIter):
        output = tf.sparse.sparse_dense_matmul(tf.sparse.slice(total_pattern, start=[matrix_size*i,0], size = [matrix_size, matrix_size]), input_mat[i])
    elapsed = time.perf_counter() - start
    #print(f"{nIter} BS Mat Mul in: {elapsed}")
    print(elapsed)
