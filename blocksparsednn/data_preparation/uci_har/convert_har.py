import os.path
import h5py
import numpy as np
from itertools import chain
from typing import List

from utils.file_utils import iterate_dir, make_dir, read_jsonl_gz
from utils.constants import INPUTS, OUTPUT


FEATURES = 6


def convert_fold(input_folder: str, output_folder: str, ndims: int):
    make_dir(output_folder)
    records = chain(*(read_jsonl_gz(data_file) for data_file in iterate_dir(input_folder, pattern='.*jsonl.gz')))
    
    inputs_lst: List[np.ndarray] = []
    output_lst: List[int] = []

    for record in records:
        inpt = np.array(record[INPUTS])
        if ndims == 2:
            inpt = inpt.reshape((1, -1, FEATURES))
        else:
            inpt = inpt.reshape(1, -1)

        inputs_lst.append(inpt)
        output_lst.append(int(record[OUTPUT]))

    inputs = np.vstack(inputs_lst)
    outputs = np.array(output_lst)
    assert inputs.shape[0] == outputs.shape[0], 'Misaligned datasets'

    print(outputs.shape)

    print('Converting {0} records'.format(inputs.shape[0]))

    with h5py.File(os.path.join(output_folder, 'data.h5'), 'w') as fout:
        input_dataset = fout.create_dataset(INPUTS, inputs.shape, dtype='f')
        input_dataset.write_direct(inputs)

        output_dataset = fout.create_dataset(OUTPUT, outputs.shape, dtype='i')
        output_dataset.write_direct(outputs)


OUTPUT_FOLDER = '../../datasets/uci_har_1d'
INPUT_FOLDER = '../../datasets/uci_har'
NDIMS = 1

make_dir(OUTPUT_FOLDER)

# Convert the data folds
for fold in ['train', 'validation', 'test']:
    convert_fold(input_folder=os.path.join(INPUT_FOLDER, fold),
                 output_folder=os.path.join(OUTPUT_FOLDER, fold),
                 ndims=NDIMS)
