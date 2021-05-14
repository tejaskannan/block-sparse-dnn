import tensorflow as tf
import numpy as np

DIMS = 4096
SPARSITY = 0.5
TRIALS = 1000

num_nonzero = int(DIMS * DIMS * SPARSITY)

# Generate the sparse matrix
rand = np.random.RandomState(seed=53)

all_indices = np.arange(DIMS)
row_indices = np.sort(rand.choice(all_indices, size=num_nonzero, replace=True)).reshape(-1, 1)
col_indices = rand.choice(all_indices, size=num_nonzero, replace=True).reshape(-1, 1)
sparse_indices = np.concatenate([row_indices, col_indices], axis=-1)  # [NNZ, 2]

sparse_values = rand.uniform(low=-2.0, high=2.0, size=num_nonzero) # [NNZ]

# Generate the input matrix
input_mat = rand.uniform(low=-2.0, high=2.0, size=(DIMS, DIMS))


with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    
    inputs = tf.compat.v1.placeholder(shape=[DIMS, DIMS],
                                      dtype=tf.float32,
                                      name='inputs')

    weights = tf.compat.v1.placeholder(shape=[num_nonzero],
                                       dtype=tf.float32,
                                       name='sp-weights')

    indices = tf.compat.v1.placeholder(shape=[num_nonzero, 2],
                                       dtype=tf.int64,
                                       name='sp-indices')


    sp_mat = tf.sparse.SparseTensor(values=weights, indices=indices, dense_shape=[DIMS, DIMS])

    output = tf.sparse.sparse_dense_matmul(sp_mat, inputs)

    sess.run(tf.compat.v1.global_variables_initializer())
    
    feed_dict = {
        inputs: input_mat,
        weights: sparse_values,
        indices: sparse_indices
    }

    start = time.time()
    for _ in range(TRIALS):
        result = sess.run(output, feed_dict=feed_dict)

    end = time.time()

    print('Sparse Matmul took {0} second.'.format(end - start))
