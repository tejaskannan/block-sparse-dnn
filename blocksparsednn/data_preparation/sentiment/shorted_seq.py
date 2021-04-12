import h5py
import os.path
from utils.constants import INPUTS, OUTPUT
from utils.file_utils import make_dir

def truncate_fold(input_file: str, output_file: str, new_seq_length: int):
    with h5py.File(input_file, 'r') as fin:
        inputs = fin[INPUTS][:, 0:new_seq_length, :]
        outputs = fin[OUTPUT][:]

    with h5py.File(output_file, 'w') as fout:
        input_ds = fout.create_dataset(INPUTS, inputs.shape, dtype='f')
        output_ds = fout.create_dataset(OUTPUT, outputs.shape, dtype='i')

        input_ds.write_direct(inputs)
        output_ds.write_direct(outputs)


input_folder = '/home/tejask/Documents/sentiment'
output_folder = '../../datasets/sentiment16'
seq_length = 16

# Make directories
make_dir(output_folder)
make_dir(os.path.join(output_folder, 'train'))
make_dir(os.path.join(output_folder, 'validation'))
make_dir(os.path.join(output_folder, 'test'))


for fold in ['train', 'validation', 'test']:
    truncate_fold(input_file=os.path.join(input_folder, fold, 'data.h5'),
                  output_file=os.path.join(output_folder, fold, 'data.h5'),
                  new_seq_length=seq_length)
