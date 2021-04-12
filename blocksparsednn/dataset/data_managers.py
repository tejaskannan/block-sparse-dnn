import os.path
import numpy as np
import h5py
from typing import Iterable, Tuple, Any

from blocksparsednn.utils.file_utils import read_jsonl_gz, iterate_dir
from blocksparsednn.utils.constants import INPUTS, OUTPUT


class DataManager:

    def __init__(self, folder: str):
        self._folder = folder
        self._is_loaded = False

    def shuffle(self):
        pass

    def iterate(self) -> Iterable[Tuple[Any, Any]]:
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def close(self):
        pass


class InMemoryDataManager(DataManager):

    def __init__(self, folder: str):
        super().__init__(folder)
        self._dataset = []
        self._idx = []
        self._rand = np.random.RandomState(seed=172)

    def load(self):
        if self._is_loaded:
            return

        for data_file in iterate_dir(self._folder, pattern='.*\.jsonl\.gz'):
            self._dataset.extend(read_jsonl_gz(data_file))            

        self._idx = np.arange(len(self._dataset))
        self._is_loaded = True

    def shuffle(self):
        assert self._is_loaded, 'Must call load() first'
        self._rand.shuffle(self._idx)

    def iterate(self) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        assert self._is_loaded, 'Must call load() first'

        for idx in self._idx:
            element = self._dataset[idx]
            yield np.array(element[INPUTS]), np.array(element[OUTPUT])

    def close(self):
        self._dataset = []


class H5DataManager(DataManager):

    def __init__(self, folder: str, load_in_memory: bool):
        super().__init__(folder)
        self._path = os.path.join(folder, 'data.h5')
        self._load_in_memory = load_in_memory
        self._file = None
        self._inputs: List[np.ndarray] = []
        self._output: List[int] = []
        self._idx: List[int] = []
        self._rand = np.random.RandomState(924)

    def load(self):
        if self._is_loaded:
            return

        self._file = h5py.File(self._path, 'r')

        if self._load_in_memory:
            self._inputs = self._file[INPUTS][:]
            self._output = self._file[OUTPUT][:]
        else:
            self._inputs = self._file[INPUTS]
            self._output = self._file[OUTPUT]

        self._idx = np.arange(self._file[INPUTS].shape[0])
        self._is_loaded = True

    def shuffle(self):
        assert self._is_loaded, 'Must call load() first'
        self._rand.shuffle(self._idx)

    def iterate(self) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        assert self._is_loaded, 'Must call load() first'      

        for idx in self._idx:
           yield self._inputs[idx], np.array(self._output[idx])

    def close(self):
        if not self._is_loaded:
            return

        self._inputs = []
        self._output = []

        self._file.close()
        self._idx = []
        self._file = None


def make_data_manager(folder: str, manager_type: str):
    manager_type = manager_type.lower()

    if manager_type == 'memory':
        return H5DataManager(folder, load_in_memory=True)
    elif manager_type == 'disk':
        return H5DataManager(folder, load_in_memory=False)
    else:
        raise ValueError('Unknown Data Manager Type: {0}'.format(manager_type))
