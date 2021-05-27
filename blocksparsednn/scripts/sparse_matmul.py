import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import time
import gzip
import codecs
import json

from enum import Enum, auto
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops as sp_ops
from itertools import product
from typing import Tuple, Dict, Iterable, Any

from blocksparsednn.layers.block_sparse_layer import BlockSparseLayer


MIN_VAL = -2
MAX_VAL = 2


def save_jsonl_gz(data: Iterable[Any], file_path: str) -> None:
    assert file_path.endswith('.jsonl.gz'), 'Must provide a json lines gzip file.'

    with gzip.GzipFile(file_path, 'wb') as f:
        writer = codecs.getwriter('utf-8')
        for element in data:
            writer(f).write(json.dumps(element))
            writer(f).write('\n')


class MatMulMode(Enum):
    DENSE = auto()
    COO = auto()
    BLOCK = auto()


class Multiplier:

    def __init__(self, n: int, sparsity: float, mode: MatMulMode, block_size: int):
        self._sess = tf.Session(graph=tf.Graph())
        self._is_sparse = mode in (MatMulMode.COO, MatMulMode.BLOCK)
        self._sparsity = sparsity  # Fraction of nonzero values
        self._nnz = int(sparsity * n * n)

        with self._sess.graph.as_default():
            self._dense_input = tf.placeholder(dtype=tf.float32, shape=[n, n], name='dense-input')
            self._output = tf.placeholder(dtype=tf.float32, shape=[n, n], name='output')

            # Create the weights variable
            if mode == MatMulMode.COO:
                weights = tf.get_variable(shape=(self._nnz,),
                                                          initializer=tf.glorot_uniform_initializer(),
                                                          dtype=tf.float32,
                                                          name='weights')
 
                
                indices = make_sparse_indices(n, n, nonzero_frac=sparsity)

                sparse_mat = tf.SparseTensor(indices=indices, values=weights, dense_shape=(n, n))
                result = tf.sparse.sparse_dense_matmul(sparse_mat, self._dense_input)
            elif mode == MatMulMode.BLOCK:
                d = int(n / block_size)
                pattern = (np.random.uniform(low=0.0, high=1.0, size=(d, d)) < sparsity).astype(float)

                layer = BlockSparseLayer(pattern=pattern,
                                         units=n,
                                         block_size=block_size,
                                         initializer=tf.glorot_uniform_initializer(),
                                         dtype=tf.float32)
                result = layer(self._dense_input)
            else:
                weights = tf.get_variable(shape=(n, n),
                                                    initializer=tf.glorot_uniform_initializer(),
                                                    dtype=tf.float32,
                                                    name='weights')
                result = tf.matmul(self._dense_input, weights)

            self._result = result
            self._loss = tf.reduce_sum(tf.square(self._result - self._output))

            trainable_vars = self._sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self._gradients = tf.gradients(self._loss, trainable_vars)

            self._sess.run(tf.global_variables_initializer())

    def run(self, dense_mat: np.ndarray, output: np.ndarray, should_run_loss: bool) -> float:
        with self._sess.graph.as_default():

            feed_dict = {
                self._dense_input: dense_mat,
                self._output: output
            }

            start = time.time()

            if should_run_loss:
                loss = self._sess.run(self._loss, feed_dict=feed_dict)
            else:
                grad = self._sess.run(self._gradients, feed_dict=feed_dict)

            end = time.time()

            return end - start


def make_sparse_matrix(n: int, m: int, nonzero_frac: float) -> sp.coo_matrix:
    probs = np.random.uniform(low=0.0, high=1.0, size=(n, m))
    mat = np.random.uniform(low=MIN_VAL, high=MAX_VAL, size=(n, m))
    masked_mat = mat * np.less(probs, nonzero_frac).astype(float)
    return sp.coo_matrix(masked_mat)


def make_sparse_indices(n: int, m: int, nonzero_frac: float) -> np.ndarray:
    idx = np.arange(start=0, stop=n * m)
    nnz = int(n * m * nonzero_frac)

    selected_idx = np.sort(np.random.choice(idx, size=nnz))  # [nnz]
    mat_indices = [(int(i / m), int(i % m)) for i in selected_idx]
    return np.array(mat_indices)


def make_dense_matrix(n: int, m: int) -> np.ndarray:
    return np.random.uniform(low=MIN_VAL, high=MAX_VAL, size=(n, m))


def run(n: int, nonzero_frac: float, trials: int, mode: MatMulMode, should_run_loss: bool) -> float:
    """
    Executes sparse-dense matrix multiplication on random matrices of the given size.
    """
    multiplier = Multiplier(n=n, sparsity=nonzero_frac, mode=mode, block_size=16)

    runtimes: List[float] = []
    for i in range(trials + 1):
        sp_indices = make_sparse_indices(n, n, nonzero_frac=nonzero_frac)
        dense_mat = make_dense_matrix(n, n)
        output_mat = make_dense_matrix(n, n)

        t = multiplier.run(dense_mat=dense_mat,
                           output=output_mat,
                           should_run_loss=should_run_loss)

        if i > 0:
            runtimes.append(t)

    return np.average(runtimes), np.std(runtimes)


if __name__ == '__main__':

    nonzero_fracs = [0.03125, 0.1]
    ns = [128]

    trials = 20

    results: Dict[Tuple[float, int, int, int], Dict[str, float]] = dict()

    print('Nonzero Frac | n | COO Loss | BLOCK Loss | Dense Loss | COO Grad | BLOCK Grad | Dense Grad')

    for nonzero, n in product(nonzero_fracs, ns):
        key = '{0:.2f} {1}'.format(nonzero, n)
        results[key] = dict()

        coo_loss = run(n=n, nonzero_frac=nonzero, trials=trials, mode=MatMulMode.COO, should_run_loss=True)
        block_loss = run(n=n, nonzero_frac=nonzero, trials=trials, mode=MatMulMode.BLOCK, should_run_loss=True)
        dense_loss = run(n=n, nonzero_frac=nonzero, trials=trials, mode=MatMulMode.DENSE, should_run_loss=True)

        coo_grad = run(n=n, nonzero_frac=nonzero, trials=trials, mode=MatMulMode.COO, should_run_loss=False)
        block_grad = run(n=n, nonzero_frac=nonzero, trials=trials, mode=MatMulMode.BLOCK, should_run_loss=False)
        dense_grad = run(n=n, nonzero_frac=nonzero, trials=trials, mode=MatMulMode.DENSE, should_run_loss=False)

        print('{0:.2f} & {1} & {2:.4f} & {3:.4f} & {4:.4f} & {5:.4f} & {6:.4f} & {7:.4f}'.format(nonzero, n, coo_loss[0], block_loss[0], dense_loss[0], coo_grad[0], block_grad[0], dense_grad[0]))

        # Save results in the results dict
        results[key]['coo_loss'] = coo_loss
        results[key]['block_loss'] = block_loss
        results[key]['dense_loss'] = dense_loss
        results[key]['coo_grad'] = coo_grad
        results[key]['block_grad'] = block_grad
        results[key]['dense_grad'] = dense_grad

    
    output_file = 'sparse_matmul_results.jsonl.gz'
    save_jsonl_gz([results], output_file)
