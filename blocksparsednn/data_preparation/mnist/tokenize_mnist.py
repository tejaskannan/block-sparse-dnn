import os.path
import gzip
import numpy as np
import matplotlib.pyplot as plt
import h5py
from typing import Iterable, Tuple, List

from utils.constants import INPUTS, OUTPUT
from utils.data_writer import DataWriter
from utils.file_utils import make_dir


DIM = 28
CHUNK_SIZE = 10000
SEED = 1827
TRAIN_FRAC = 0.8


def get_data(images_path: str, labels_path: str, ndims: int) -> Iterable[Tuple[np.ndarray, int]]:
    with gzip.open(images_path) as im_file, gzip.open(labels_path) as label_file:
        # Move past the headers
        im_file.read(16)
        label_file.read(8)

        # Read the ima
        is_finished = False
        while not is_finished:

            buf = im_file.read(DIM * DIM * CHUNK_SIZE)
            images = np.frombuffer(buf, dtype=np.uint8).astype(float)

            labels: List[int] = []
            for _ in range(CHUNK_SIZE):
                labels.extend(np.frombuffer(label_file.read(1), dtype=np.uint8).astype(int))

            if ndims == 2:
                newshape = (len(labels), DIM, DIM)
            else:
                newshape = (len(labels), DIM * DIM, 1)

            images = images.reshape(newshape)

            if len(labels) < CHUNK_SIZE:
                is_finished = True

            for img, label in zip(images, labels):
                yield (img, label)
    

def write_h5py_dataset(images: List[np.ndarray], labels: List[int], path: str):
    make_dir(os.path.split(path)[0])

    with h5py.File(path, 'w') as fout:
        img_dataset = np.vstack([np.expand_dims(img, axis=0) for img in images])

        inputs = fout.create_dataset(INPUTS, img_dataset.shape, dtype='f')
        inputs.write_direct(img_dataset)

        output = fout.create_dataset(OUTPUT, (len(labels), ), dtype='i')
        output.write_direct(np.array(labels))



def write_train_dataset(images_path: str, labels_path: str, output_folder: str, ndims: int):
    make_dir(output_folder)
    
    #train_writer = DataWriter(output_folder=os.path.join(output_folder, 'train'),
    #                          file_prefix='data',
    #                          chunk_size=CHUNK_SIZE)
    #val_writer = DataWriter(output_folder=os.path.join(output_folder, 'validation'),
    #                        file_prefix='data',
    #                        chunk_size=CHUNK_SIZE)

    data_iterator = get_data(images_path=images_path, labels_path=labels_path, ndims=ndims)
    rand = np.random.RandomState(seed=SEED)
    train_samples, val_samples = 0, 0

    train_images: List[np.ndarray] = []
    train_labels: List[int] = []

    val_images: List[np.ndarray] = []
    val_labels: List[np.ndarray] = []

    for idx, (img, label) in enumerate(data_iterator):
    
        r = rand.uniform()
        if r < TRAIN_FRAC:
            train_images.append(img)
            train_labels.append(label)

            # writer = train_writer
            train_samples += 1
        else:
            # writer = val_writer

            val_images.append(img)
            val_labels.append(label)

            val_samples += 1

        #sample = {
        #    INPUTS: img.tolist(),
        #    OUTPUT: int(label)
        #}

        #writer.add(sample)

        if (idx + 1) % CHUNK_SIZE == 0:
            print('Completed {0} samples.'.format(idx + 1))

    # train_writer.flush()
    # val_writer.flush()

    write_h5py_dataset(images=train_images, labels=train_labels, path=os.path.join(output_folder, 'train', 'data.h5'))
    write_h5py_dataset(images=val_images, labels=val_labels, path=os.path.join(output_folder, 'validation', 'data.h5'))
    print('Finished. {0} training samples, {1} validation samples.'.format(train_samples, val_samples))


def write_test_dataset(images_path: str, labels_path: str, output_folder: str, ndims: int):
    make_dir(output_folder)

    # with DataWriter(os.path.join(output_folder, 'test'), file_prefix='data', chunk_size=CHUNK_SIZE) as writer:
    data_iterator = get_data(images_path=images_path, labels_path=labels_path, ndims=ndims)
    test_sample_count = 0

    test_images: List[np.ndarray] = []
    test_labels: List[int] = []

    for img, label in data_iterator:
       # sample = {
       #     INPUTS: img.tolist(),
       #     OUTPUT: int(label)
       # }

       # writer.add(sample)
        test_images.append(img)
        test_labels.append(label)
        test_sample_count += 1

        if test_sample_count % CHUNK_SIZE == 0:
            print('Completed {0} samples.'.format(test_sample_count))

    write_h5py_dataset(images=test_images, labels=test_labels, path=os.path.join(output_folder, 'test', 'data.h5'))

    print('Finished. {0} testing samples'.format(test_sample_count))


WRITE_TRAIN = True
WRITE_TEST = True
NDIMS = 2

if WRITE_TRAIN:
    write_train_dataset(images_path='../../datasets/mnist_2d/raw/train-images.gz',
                        labels_path='../../datasets/mnist_2d/raw/train-labels.gz',
                        output_folder='../../datasets/mnist2d',
                        ndims=NDIMS)

if WRITE_TEST:
    write_test_dataset(images_path='../../datasets/mnist_2d/raw/test-images.gz',
                       labels_path='../../datasets/mnist_2d/raw/test-labels.gz',
                       output_folder='../../datasets/mnist2d',
                       ndims=NDIMS)
