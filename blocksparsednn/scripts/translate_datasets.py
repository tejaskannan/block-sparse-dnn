"""
Translates the testing set to a format easier read by the C implementation.
This allows us to perform easier debugging.
"""
import h5py
import os.path
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler

from blocksparsednn.conversion.convert_utils import array_to_fixed_point


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--precision', type=int, required=True)
    args = parser.parse_args()

    # Set the base path
    base = os.path.join('..', 'datasets', args.dataset)

    # Load the training set and fit the scaler
    with h5py.File(os.path.join(base, 'train', 'data.h5'), 'r') as fin:
        train_inputs = fin['inputs'][:]
    
    scaler = StandardScaler().fit(train_inputs)

    # Load the testing set
    with h5py.File(os.path.join(base, 'test', 'data.h5'), 'r') as fin:
        test_inputs = fin['inputs'][:]
        test_outputs = fin['output'][:]

    test_inputs = scaler.transform(test_inputs)

    # Write the test inputs to a text file
    with open(os.path.join(base, 'test_inputs.txt'), 'w') as fout:
        for features in test_inputs:
            quantized = array_to_fixed_point(features, precision=args.precision, width=16)
            feature_string = ','.join(map(str, quantized))

            fout.write(feature_string)
            fout.write('\n')

    # Write the test output
    with open(os.path.join(base, 'test_labels.txt'), 'w') as fout:
        for label in test_outputs:
            fout.write(str(int(label)))
            fout.write('\n')
