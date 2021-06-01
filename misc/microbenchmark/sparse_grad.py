import tensorflow as tf
import numpy as np
import time

from argparse import ArgumentParser
from blocksparsednn.utils.file_utils import save_jsonl_gz
from typing import List, Dict, Any


def run_benchmark(hidden_size: int, sparsity: float, trials: int) -> Dict[str, float]:
    times: List[float] = []

    rand = np.random.RandomState(seed=51)

    # Create a (random) sparsity pattern
    indices = np.arange(hidden_size * hidden_size)
    nonzero_indices = rand.choice(indices, size=int(round(sparsity * hidden_size * hidden_size)), replace=False)
    num_nonzero = len(nonzero_indices)

    rows: List[int] = []
    cols: List[int] = []
    for idx in nonzero_indices:
        row = int(idx / hidden_size)
        col = int(idx % hidden_size)

        rows.append(row)
        cols.append(col)

    sparse_indices: List[List[int]] = [[r, c] for r, c in sorted(zip(rows, cols))]

    with tf.Session(graph=tf.Graph()) as sess:
 
        inputs = tf.placeholder(shape=(hidden_size, hidden_size), dtype=tf.float32, name='inputs')


        weights = tf.get_variable(shape=(num_nonzero, ), name='W',dtype=tf.float32)
        sp_weights = tf.SparseTensor(values=weights,
                                     indices=sparse_indices,
                                     dense_shape=(hidden_size, hidden_size))

        transformed = tf.sparse.sparse_dense_matmul(sp_weights, inputs)
        grad = tf.gradients(transformed, weights)

        sess.run(tf.global_variables_initializer())

        for i in range(trials + 1):
            mat = rand.uniform(low=-2.0, high=2.0, size=(hidden_size, hidden_size))

            start = time.perf_counter()
            sess.run(grad, feed_dict={inputs: mat})
            end = time.perf_counter()

            elapsed = end - start

            if i > 0:
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
    parser.add_argument('--sparsity', type=float, required=True)
    parser.add_argument('--trials', type=int, required=True)
    args = parser.parse_args()

    results: Dict[str, Any] = {
        'type': 'sparse_gradients',
        'hidden-size': args.hidden_size,
        'sparsity': args.sparsity,
        'trials': args.trials,
    }

    bench_results = run_benchmark(hidden_size=args.hidden_size,
                                  sparsity=args.sparsity,
                                  trials=args.trials)
    results['benchmark'] = bench_results

    output_path = 'sparse_grad_{0}.jsonl.gz'.format(args.hidden_size)
    save_jsonl_gz([results], output_path)
