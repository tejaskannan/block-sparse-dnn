import tensorflow as tf
import numpy as np
import time

from argparse import ArgumentParser
from blocksparsednn.utils.file_utils import save_jsonl_gz
from typing import List, Dict, Any


MARGIN = 5


def run_benchmark(hidden_size: int, sparsity: float, trials: int) -> Dict[str, float]:
    times: List[float] = []

    rand = np.random.RandomState(seed=51)

    with tf.Session(graph=tf.Graph()) as sess:
 
        inputs = tf.placeholder(shape=(hidden_size, hidden_size), dtype=tf.float32, name='inputs')
        weights = tf.get_variable(shape=(hidden_size, hidden_size), name='W',dtype=tf.float32)

        transformed = tf.matmul(inputs, weights)

        grad = tf.gradients(transformed, weights)

        sess.run(tf.global_variables_initializer())

        for i in range(trials + MARGIN):
            mat = rand.uniform(low=-2.0, high=2.0, size=(hidden_size, hidden_size))
            feed_dict = {inputs: mat}

            start = time.perf_counter()
            sess.run(grad, feed_dict=feed_dict)
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
    parser.add_argument('--sparsity', type=float, required=True)
    parser.add_argument('--trials', type=int, required=True)
    args = parser.parse_args()

    results: Dict[str, Any] = {
        'type': 'dense_gradients',
        'hidden-size': args.hidden_size,
        'sparsity': args.sparsity,
        'trials': args.trials,
    }

    bench_results = run_benchmark(hidden_size=args.hidden_size,
                                  sparsity=args.sparsity,
                                  trials=args.trials)
    results['benchmark'] = bench_results

    output_path = 'dense_grad_{0}.jsonl.gz'.format(args.hidden_size)
    save_jsonl_gz([results], output_path)
