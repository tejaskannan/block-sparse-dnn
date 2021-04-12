import h5py
import numpy as np
import os.path
from typing import List

from blocksparsednn.utils.constants import INPUTS, OUTPUT
from blocksparsednn.utils.file_utils import make_dir


NUM_TRAIN = 1000
NUM_VAL = 200
NUM_TEST  = 200
NUM_FEATURES = 6


def generate_data(num_samples: int, rand: np.random.RandomState, output_file: str):

    random_vals = rand.uniform(low=-5.0, high=5.0, size=(num_samples, NUM_FEATURES))  # [N, D]

    features_list: List[np.ndarray] = []
    labels_list: List[int] = []

    for vals in random_vals:
        label = rand.uniform(low=0.0, high=1.0) < 0.5
        
        if label == 0:
            features = np.square(vals)
        else:
            features = np.sin(vals)

        features_list.append(np.expand_dims(features, axis=0))
        labels_list.append(label)

    inputs = np.vstack(features_list)
    labels = np.vstack(labels_list).reshape(-1)

    with h5py.File(output_file, 'w') as fout:
        input_dataset = fout.create_dataset(INPUTS, shape=inputs.shape, dtype='f')
        input_dataset.write_direct(inputs)

        output_dataset = fout.create_dataset(OUTPUT, shape=labels.shape, dtype='i')
        output_dataset.write_direct(labels)

    return np.vstack(features_list), labels


if __name__ == '__main__':
    base = os.path.join('..', '..', 'datasets', 'function')
    make_dir(base)

    rand = np.random.RandomState(seed=2316)

    train_folder = os.path.join(base, 'train')
    make_dir(train_folder)
    generate_data(NUM_TRAIN, rand=rand, output_file=os.path.join(train_folder, 'data.h5'))

    val_folder = os.path.join(base, 'validation')
    make_dir(val_folder)
    generate_data(NUM_VAL, rand=rand, output_file=os.path.join(val_folder, 'data.h5'))

    test_folder = os.path.join(base, 'test')
    make_dir(test_folder)
    generate_data(NUM_VAL, rand=rand, output_file=os.path.join(test_folder, 'data.h5'))
