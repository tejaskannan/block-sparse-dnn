import h5py
import os.path
import numpy as np

dataset_name = 'pavement'
fold = 'validation'

with h5py.File(os.path.join('..', 'datasets', dataset_name, fold, 'data.h5'), 'r') as fin:
    inputs = fin['inputs'][:]
    labels = fin['output'][:]

print('Input Shape: {0}'.format(inputs.shape))
print('# Labels: {0}'.format(np.max(labels) + 1))
