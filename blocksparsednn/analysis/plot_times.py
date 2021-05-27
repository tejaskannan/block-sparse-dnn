import matplotlib.pyplot as plt
import numpy as np


DATASETS = ['Pen Digits', 'Seizure', 'Tiselac', 'Fashion MNIST', 'HAR']

DENSE = np.array([21.93, 34.2, 245.12, 207.11, 72.986])
SPARSE = np.array([36.02, 54.8, 381.16, 285, 136.788])
BLOCK_DIAG = np.array([25.86, 39.66, 301, 256, 99.58])

WIDTH = 0.25


if __name__ == '__main__':
    
    normalized_dense = SPARSE / DENSE
    normalized_sparse = SPARSE / SPARSE
    normalized_block = SPARSE / BLOCK_DIAG

    fig, ax = plt.subplots(figsize=(8, 6))

    xs = np.arange(len(DATASETS))

    ax.bar(xs - WIDTH, normalized_dense, width=WIDTH, label='Dense', color='#7FC97F')
    ax.bar(xs, normalized_sparse, width=WIDTH, label='Sparse', color='#BEAED4')
    ax.bar(xs + WIDTH, normalized_block, width=WIDTH, label='Hybrid Block Diag', color='#386CB0')

    ax.legend(fontsize=12, framealpha=0.0)

    ax.set_xticks(ticks=xs)
    ax.set_xticklabels(DATASETS, size=14)

    ax.set_xlabel('Dataset', fontsize=14)
    ax.set_ylabel('Speedup over Sparse', fontsize=14)

    ax.set_title('Normalized Training Time', fontsize=16)

    plt.savefig('train_time_32.png', transparent=True, bbox_inches='tight')

    dense_speedup = np.average(normalized_dense)
    sparse_speedup = np.average(normalized_sparse)
    block_speedup = np.average(normalized_block)

    print('Avg Speedup: Dense -> {0:.4f}, Sparse -> {1:.4f}, Block Diagonal -> {2:.4f}'.format(dense_speedup, sparse_speedup, block_speedup))
