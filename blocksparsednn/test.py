import os.path
from argparse import ArgumentParser
from typing import Optional

from blocksparsednn.neural_network.model_factory import get_neural_network
from blocksparsednn.dataset.dataset import Dataset
from blocksparsednn.utils.constants import MODEL_FILE_FMT, TEST_LOG_FMT, HYPERS_FILE_FMT, FLOPS
from blocksparsednn.utils.file_utils import read_pickle_gz, save_jsonl_gz, extract_model_name


def test(model_file_name: str, dataset: Dataset, save_folder: str, batch_size: Optional[int]):
    # Create the save folder with the data name and model name
    model_name = model_file_name.split('-')[0]
    save_folder = os.path.join(save_folder, dataset.name, model_name)

    # Restore the model
    model_path = os.path.join(save_folder, MODEL_FILE_FMT.format(model_file_name))

    hypers_path = os.path.join(save_folder, HYPERS_FILE_FMT.format(model_file_name))
    hypers = read_pickle_gz(hypers_path)

    name = hypers['name'].lower()
    model_cls = get_neural_network(name)

    # Count the FLOPS
    frozen_model = model_cls.restore(model_path, is_frozen=True)
    flops = frozen_model.count_flops()

    # Create the unfrozen model for testing
    model = model_cls.restore(model_path, is_frozen=False)

    # Execute the model on the test set
    test_results = model.test(dataset=dataset, batch_size=batch_size)
    test_results[FLOPS] = flops

    # Save the testing results
    test_log_path = os.path.join(save_folder, TEST_LOG_FMT.format(model_file_name))
    save_jsonl_gz([test_results], test_log_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-file', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion_mnist', 'uci_har', 'function'], required=True)
    parser.add_argument('--batch-size', type=int)
    args = parser.parse_args()

    # Get the model file components
    save_folder, file_name = os.path.split(args.model_file)
    model_name = extract_model_name(file_name)

    # Make the data object
    dataset_folder = os.path.join('datasets', args.dataset)
    dataset = Dataset(folder=dataset_folder, dataset_type='memory')

    test(model_file_name=model_name,
         dataset=dataset,
         save_folder=save_folder,
         batch_size=args.batch_size)
