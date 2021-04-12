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


def get_data(images_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray]:
    # Get the labels
    with gzip.open(labels_path) as label_file:
        label_buffer = np.frombuffer(label_file.read(), dtype=np.uint8, offset=8)
        labels = label_buffer.astype(int)

    # Get the images
    with gzip.open(images_path) as images_file:
        images_buffer = np.frombuffer(images_file.read(), dtype=np.uint8, offset=16)
        images = images_buffer.astype(float).reshape(len(labels), DIM, DIM)  # [N, 28, 28]
        images = images / 255.0  # Normalize into [0, 1]

    assert images.shape[0] == labels.shape[0], 'Misaligned images and labels'

    return images, labels


def write_h5py_dataset(images: np.ndarray, labels: np.ndarray, path: str):
    make_dir(os.path.split(path)[0])

    assert images.shape[0] == labels.shape[0], 'Misaligned images and labels'

    with h5py.File(path, 'w') as fout:
        inputs = fout.create_dataset(INPUTS, images.shape, dtype='f')
        inputs.write_direct(images)

        output = fout.create_dataset(OUTPUT, (len(labels), ), dtype='i')
        output.write_direct(labels)


def write_train_dataset(images_path: str, labels_path: str, output_folder: str):
    make_dir(output_folder)

    images, labels = get_data(images_path=images_path, labels_path=labels_path)
    rand = np.random.RandomState(seed=SEED)
    train_samples, val_samples = 0, 0

    train_images: List[np.ndarray] = []
    train_labels: List[int] = []

    val_images: List[np.ndarray] = []
    val_labels: List[np.ndarray] = []

    for idx, (img, label) in enumerate(zip(images, labels)):
    
        r = rand.uniform()
        if r < TRAIN_FRAC:
            train_images.append(np.expand_dims(img, axis=0))
            train_labels.append(label)

            train_samples += 1
        else:
            val_images.append(np.expand_dims(img, axis=0))
            val_labels.append(label)

            val_samples += 1

        if (idx + 1) % CHUNK_SIZE == 0:
            print('Completed {0} samples.'.format(idx + 1))

    write_h5py_dataset(images=np.vstack(train_images),
                       labels=np.array(train_labels),
                       path=os.path.join(output_folder, 'train', 'data.h5'))
    write_h5py_dataset(images=np.vstack(val_images),
                       labels=np.array(val_labels),
                       path=os.path.join(output_folder, 'validation', 'data.h5'))
    
    print('Finished. {0} training samples, {1} validation samples.'.format(train_samples, val_samples))


def write_test_dataset(images_path: str, labels_path: str, output_folder: str):
    make_dir(output_folder)

    images, labels = get_data(images_path=images_path, labels_path=labels_path)
    test_sample_count = images.shape[0]

    write_h5py_dataset(images=images, labels=labels, path=os.path.join(output_folder, 'test', 'data.h5'))
    print('Finished. {0} test samples.'.format(test_sample_count))


WRITE_TRAIN = True
WRITE_TEST = True
BASE = '../../datasets/mnist'

if WRITE_TRAIN:
    write_train_dataset(images_path=os.path.join(BASE, 'raw/train-images.gz'),
                        labels_path=os.path.join(BASE, 'raw/train-labels.gz'),
                        output_folder=BASE)

if WRITE_TEST:
    write_test_dataset(images_path=os.path.join(BASE, 'raw/test-images.gz'),
                       labels_path=os.path.join(BASE, 'raw/test-labels.gz'),
                       output_folder=BASE)
