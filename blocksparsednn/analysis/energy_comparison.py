import csv
import os.path
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict

from blocksparsednn.utils.file_utils import iterate_dir


NUM_SAMPLES = 15
BLOCK_DIAG = 'block_diag'
SPARSE = 'sparse'
DENSE = 'dense'


def get_energy(input_path: str) -> float:
    energy = 0.0

    with open(input_path, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')

        for idx, line in enumerate(reader):
            if idx > 0:
                energy = float(line[-1])  # Energy in uJ

    return energy / 1000.0


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    args = parser.parse_args()

    # Get the results from bluetooth alone
    bt_energy = get_energy(os.path.join(args.input_folder, 'bluetooth_alone.csv'))

    energy_results = defaultdict(list)

    for path in iterate_dir(args.input_folder, pattern='.*csv'):
        energy = get_energy(path) # Energy in mJ
        processing_energy = (energy - bt_energy) / NUM_SAMPLES

        file_name = os.path.basename(path)
        if file_name.startswith(BLOCK_DIAG):
            energy_results[BLOCK_DIAG].append(processing_energy)
        elif file_name.startswith(SPARSE):
            energy_results[SPARSE].append(processing_energy)
        elif file_name.startswith(DENSE):
            energy_results[DENSE].append(processing_energy)

    # Aggregate the results
    sparse_avg = np.average(energy_results[SPARSE])
    sparse_std = np.std(energy_results[SPARSE])

    dense_avg = np.average(energy_results[DENSE])
    dense_std = np.std(energy_results[DENSE])

    block_avg = np.average(energy_results[BLOCK_DIAG])
    block_std = np.std(energy_results[BLOCK_DIAG])

    # Format the results in a table
    print('Model & Energy (mJ) / Inference \\\\')
    print('\\midrule')
    print('Sparse & {0:.4f} (\\pm {1:.4f}) \\\\'.format(sparse_avg, sparse_std))
    print('Dense & {0:.4f} (\\pm {1:.4f}) \\\\'.format(dense_avg, dense_std))
    print('Block Diagonal & {0:.4f} (\\pm {1:.4f}) \\\\'.format(block_avg, block_std))
