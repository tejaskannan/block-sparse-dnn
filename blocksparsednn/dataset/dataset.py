import os
import numpy as np
from enum import Enum, auto
from collections import namedtuple
from typing import List, Dict, Iterable, Tuple, Any

from blocksparsednn.utils.constants import TRAIN, VAL, TEST, INPUTS
from blocksparsednn.dataset.data_managers import DataManager, make_data_manager


Batch = namedtuple('Batch', ['inputs', 'output'])


class DataSeries(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


class Dataset:

    def __init__(self, folder: str, dataset_type: str):
        self._dataset_folder = folder

        tokens = self._dataset_folder.split(os.sep)
        self._dataset_name = tokens[-1] if len(tokens[-1]) > 0 else tokens[-2]

        self._data_folders = {
            DataSeries.TRAIN: os.path.join(folder, TRAIN),
            DataSeries.VAL: os.path.join(folder, VAL),
            DataSeries.TEST: os.path.join(folder, TEST)
        }

        self._dataset_type = dataset_type

        self._data_managers: Dict[DataSeries, DataManager] = dict()
        for series, folder in self._data_folders.items():
            self._data_managers[series] = make_data_manager(folder=folder,
                                                            manager_type=dataset_type)

    @property
    def dataset_folder(self):
        return self._dataset_folder

    @property
    def name(self) -> str:
        return self._dataset_name

    def iterate_series(self, series: DataSeries) -> Iterable[Tuple[Any, Any]]:
        self._data_managers[series].load()
        return self._data_managers[series].iterate()

    def minibatch_generator(self, series: DataSeries, batch_size: int, should_shuffle: bool) -> Iterable[Batch]:
        """
        Generates mini-batches of the given size.
        """
        # Load the data series
        self._data_managers[series].load()

        if should_shuffle:
            self._data_managers[series].shuffle()

        # Lists to keep track of data in the current batch
        batch_inputs: List[np.ndarray] = []
        batch_output: List[np.ndarray] = []

        for inputs, output in self.iterate_series(series):
            batch_inputs.append(inputs)
            batch_output.append(output)

            if len(batch_inputs) >= batch_size:
                yield Batch(inputs=np.concatenate([np.expand_dims(arr, axis=0) for arr in batch_inputs]),
                            output=np.concatenate([np.expand_dims(arr, axis=0) for arr in batch_output]))

                batch_inputs = []
                batch_output = []

        # Emit the final batch
        if len(batch_inputs) == batch_size:
            yield Batch(inputs=np.concatenate([np.expand_dims(arr, axis=0) for arr in batch_inputs]),
                        output=np.concatenate([np.expand_dims(arr, axis=0) for arr in batch_output]))
