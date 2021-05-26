import numpy as np
import os.path

from argparse import ArgumentParser
from scipy.stats import ttest_ind
from typing import List, Dict, Any

from blocksparsednn.utils.file_utils import read_jsonl_gz


def compare(sparse_results: Dict[str, Any], dense_results: Dict[str, Any], block_results: Dict[str, Any]):
    # Compute aggregate statistics
    sparse_avg = np.average(sparse_results['elapsed'])
    sparse_std = np.std(sparse_results['elapsed'])

    dense_avg = np.average(dense_results['elapsed'])
    dense_std = np.std(dense_results['elapsed'])

    block_avg = np.average(block_results['elapsed'])
    block_std = np.std(block_results['elapsed'])

    # Compute the Slowdown
    sparse_slowdown = sparse_avg / block_avg
    dense_slowdown = dense_avg / block_avg
    block_slowdown = block_avg / block_avg

    # Extract the accuracy
    sparse_acc = sparse_results['accuracy']
    dense_acc = dense_results['accuracy']
    block_acc = block_results['accuracy']

    # Run the t-tests
    sparse_t_stat, sparse_p_val = ttest_ind(a=sparse_results['elapsed'], b=block_results['elapsed'], equal_var=False)
    dense_t_stat, dense_p_val = ttest_ind(a=dense_results['elapsed'], b=block_results['elapsed'], equal_var=False)

    # Print the table
    print('Model & Accuracy & Avg Time (Std) & p Value & Slowdown \\\\')
    print('\\midrule')
    print('Dense & {0:.4f} & {1:.4f} (\\pm {2:.4f}) & {3:.4f} & {4:.4f}\\\\'.format(dense_acc, dense_avg, dense_std, dense_p_val, dense_slowdown))
    print('Sparse & {0:.4f} & {1:.4f} (\\pm {2:.4f}) & {3:.4f} & {4:.4f}\\\\'.format(sparse_acc, sparse_avg, sparse_std, sparse_p_val, sparse_slowdown))
    print('Block Diagonal & {0:.4f} & {1:.4f} (\\pm {2:.4f}) & N/A & {3:.4f}\\\\'.format(block_acc, block_avg, block_std, block_slowdown))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    args = parser.parse_args()

    # Extract the results from each model type
    sparse_path = os.path.join(args.input_folder, 'sparse.jsonl.gz')
    sparse_results = list(read_jsonl_gz(sparse_path))[0]

    dense_path = os.path.join(args.input_folder, 'dense.jsonl.gz')
    if os.path.exists(dense_path):
        dense_results = list(read_jsonl_gz(dense_path))[0]
    else:
        dense_results = {
            'elapsed': [0.0],
            'accuracy': 0.0
        }

    block_diag_path = os.path.join(args.input_folder, 'block_diag.jsonl.gz')
    block_results = list(read_jsonl_gz(block_diag_path))[0]
    
    # Generate a table of results
    compare(sparse_results=sparse_results,
            dense_results=dense_results,
            block_results=block_results)
