import tensorflow as tf
import numpy as np
import time

from argparse import ArgumentParser
from blocksparse.matmul import BlocksparseMatMul
from blocksparsednn.utils.file_utils import save_jsonl_gz
from typing import List, Dict, Any


MARGIN = 5


def run_benchmark(hidden_size: int, block_size: int, sparsity: float, trials: int) -> Dict[str, float]:
    dims = int(hidden_size / block_size)

    # Create a (random) sparsity pattern
    rand = np.random.RandomState(seed=51)
    indices = np.arange(dims * dims)
    block_indices = rand.choice(indices, size=int(round(sparsity * dims * dims)), replace=False)

    pattern = np.zeros(shape=(dims, dims))
    for idx in block_indices:
        row = int(idx / dims)
        col = int(idx % dims)

        pattern[row, col] = 1

    times: List[float] = []

    with tf.Session(graph=tf.Graph()) as sess:
 
        bsmm = BlocksparseMatMul(pattern, block_size=block_size)
   
        inputs = tf.placeholder(shape=(hidden_size, hidden_size), dtype=tf.float32, name='inputs')
        weights = tf.get_variable(name='W', shape=bsmm.w_shape, dtype=tf.float32)

        transformed = bsmm(inputs, weights)

        sess.run(tf.global_variables_initializer())

        for i in range(trials + MARGIN):
            mat = rand.uniform(low=-2.0, high=2.0, size=(hidden_size, hidden_size))
            feed_dict = {inputs: mat}

            start = time.perf_counter()
            sess.run(transformed, feed_dict=feed_dict)
            end = time.perf_counter()

            elapsed = end - start

            if i >= MARGIN:
                times.append(elapsed)

    return {
        'avg': float(np.average(times)),
        'std': float(np.std(times)),
        'max': float(np.max(times)),
        'min': float(np.min(times)),
        'first_quartile': float(np.percentile(times, 25)),
        'third_quartile': float(np.percentile(times, 75)),
        'median': float(np.median(times))
    }


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--hidden-size', type=int, required=True)
    parser.add_argument('--block-sizes', type=int, required=True, nargs='+')
    parser.add_argument('--sparsity', type=float, required=True)
    parser.add_argument('--trials', type=int, required=True)
    args = parser.parse_args()

    results: Dict[str, Any] = {
        'type': 'block_sparse_matmul',
        'hidden-size': args.hidden_size,
        'block-sizes': args.block_sizes,
        'sparsity': args.sparsity,
        'trials': args.trials,
    }

    for block_size in args.block_sizes:
        block_results = run_benchmark(hidden_size=args.hidden_size,
                                      block_size=block_size,
                                      sparsity=args.sparsity,
                                      trials=args.trials)

        key = 'benchmark_{0}'.format(block_size)
        results[key] = block_results

    output_path = 'block_sparse_matmul_{0}.jsonl.gz'.format(args.hidden_size)
    save_jsonl_gz([results], output_path)
