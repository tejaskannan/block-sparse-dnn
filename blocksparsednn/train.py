import tensorflow as tf
import numpy as np
import os.path
from argparse import ArgumentParser
from typing import Dict, Any

from blocksparsednn.neural_network.model_factory import get_neural_network
from blocksparsednn.dataset.dataset import Dataset
from blocksparsednn.utils.constants import MODEL_FILE_FMT, HYPERS_FILE_FMT
from blocksparsednn.utils.file_utils import read_json, read_pickle_gz, save_jsonl_gz, iterate_dir
from blocksparsednn.test import test


def train(hypers: Dict[str, Any], save_folder: str, dataset: Dataset, should_print: bool, log_device: bool) -> str:
    # Create the neural network
    name = hypers['name'].lower()
    model_cls = get_neural_network(name)
    model = model_cls(name=name, hypers=hypers, log_device=log_device)

    # Train the model
    model_file = model.train(dataset=dataset, save_folder=save_folder, should_print=should_print)

    return model_file


if __name__ == '__main__':
    parser = ArgumentParser('Training Script for Neural Networks')
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'function', 'uci_har', 'fashion_mnist'])
    parser.add_argument('--hypers-file', type=str, required=True)
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--log-device', action='store_true')
    parser.add_argument('--should-print', action='store_true')
    args = parser.parse_args()

    if os.path.isdir(args.hypers_file):
        hypers_files = list(iterate_dir(args.hypers_file, '.*json'))
    else:
        hypers_files = [args.hypers_file]

    device = 'CPU:0' if not args.use_gpu else 'GPU:0'
    dataset_folder = os.path.join('datasets', args.dataset)

    for hypers_file in hypers_files:
        # Parse the hyper-parameters
        hypers = read_json(hypers_file)

        # Make the data object
        dataset = Dataset(folder=dataset_folder, dataset_type='memory')

        # Train the model
        with tf.device(device):
            model_file_name = train(hypers=hypers,
                                    save_folder='saved_models',
                                    dataset=dataset,
                                    should_print=args.should_print,
                                    log_device=args.log_device)

            # Test the model
            test(model_file_name=model_file_name,
                 dataset=dataset,
                 save_folder='saved_models',
                 batch_size=None)
