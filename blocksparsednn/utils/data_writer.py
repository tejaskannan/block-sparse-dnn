import os
import re

from enum import Enum, auto
from typing import List, Any, Iterable

from utils.file_utils import save_jsonl_gz, make_dir


class WriteMode(Enum):
    WRITE = auto()
    APPEND = auto()


class DataWriter:

    def __init__(self, output_folder: str, file_prefix: str, chunk_size: int, mode: str = 'w'):
        self._output_folder = output_folder
        self._file_prefix = file_prefix
        self._chunk_size = chunk_size

        # Initialize the data list
        self._dataset: List[Any] = []

        # Create the output directory if necessary
        make_dir(self._output_folder)

        # Set the writing mode
        mode = mode.lower()
        if mode in ('w', 'write'):
            self._mode = WriteMode.WRITE
        elif mode in ('a', 'append'):
            self._mode = WriteMode.APPEND
        else:
            raise ValueError('Unknown writing mode: {0}'.format(mode))

        # Set the initial file index
        self._file_index = 0
        if self._mode == WriteMode.APPEND:
            # Regex to extract index from existing files
            file_name_regex = re.compile('{0}([0-9]+)\.jsonl.gz'.format(file_prefix))

            # Get index from all existing files
            for file_name in os.listdir(output_folder):
                match = file_name_regex.match(file_name)
                if match is not None:
                    index = int(match.group(1))
                    self._file_index = max(self._file_index, index + 1)

    @property
    def output_folder(self) -> str:
        return self._output_folder

    @property
    def file_prefix(self) -> str:
        return self._file_prefix
    
    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def file_index(self) -> int:
        return self._file_index

    def current_output_file(self) -> str:
        file_name = '{0}{1:03d}.jsonl.gz'.format(self.file_prefix, self.file_index)
        return os.path.join(self.output_folder, file_name)

    def increment_file_index(self):
        self._file_index += 1

    def add(self, data: Any):
        self._dataset.append(data)
        if len(self._dataset) >= self.chunk_size:
            self.flush()

    def add_many(self, data: Iterable[Any]):
        for element in data:
            self.add(data)

    def flush(self):
        # Skip empty datasets
        if len(self._dataset) == 0:
            return

        save_jsonl_gz(self._dataset, self.current_output_file())
        self._dataset = []  # Reset the data list
        self.increment_file_index()

    def close(self):
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
