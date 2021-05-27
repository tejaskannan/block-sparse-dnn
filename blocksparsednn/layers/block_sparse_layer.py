import tensorflow as tf
import numpy as np
from typing import Any


class BlockSparseLayer:

    def __init__(self,
                 pattern: np.ndarray,
                 block_size: int,
                 units: int,
                 initializer: tf.keras.initializers.Initializer,
                 dtype: Any):
        self._pattern = pattern
        self._block_size = block_size  # B
        self._units = units  # D
        self._nnz = int(np.sum(pattern))
        self._dtype = dtype

        assert self._nnz > 0, 'Must have at least one nonzero element.'

        self._weights: List[tf.Variable] = []
        self._output_indices: List[int] = []

        input_indices = []
        for r in range(self._pattern.shape[0]):
            for c in range(self._pattern.shape[1]):
                if np.isclose(self._pattern[r, c], 1):
                    # Create the weight variable
                    w = tf.get_variable('w-{0}-{1}'.format(r, c),
                                                  shape=[block_size, block_size],
                                                  dtype=dtype,
                                                  initializer=initializer,
                                                  trainable=True)
                    self._weights.append(w)

                    # Add the feature index
                    input_indices.append(r * block_size)

                    # Add the output indices
                    self._output_indices.extend((c * block_size + i) for i in range(block_size))

        self._input_indices: tf.Variable = tf.Variable(input_indices, trainable=False)  # [NNZ]
        self._blocks: tf.Tensor = tf.stack(values=self._weights)  # [NNZ, B, B]

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Executes the block sparse matrix multiplication on the given inputs.

        Args:
            inputs: A [N, D] tensor containing features (D) for each batch element (N)
        Returns:
            A [N, D] tensor after applying the block sparse matrix.
        """
        block_results: List[tf.Tensor] = []

        dim_idx = tf.range(start=0, limit=tf.shape(inputs)[-1])  # [D]

        for i in range(self._nnz):
            start_idx = self._input_indices[i]
            # w = self._blocks[i]  # [B, B]
            w = self._weights[i]  # [B, B]
            
            input_slice = tf.slice(inputs, begin=[0, start_idx], size=[-1, self._block_size])  # [N, B]
            
            mul = tf.linalg.matmul(w, input_slice, transpose_a=True, transpose_b=True) # [B, N]
            block_results.append(mul)

        mul_results = tf.concat(block_results, axis=0) # [B * NNZ, N]

        merged = tf.math.unsorted_segment_sum(mul_results,
                                              segment_ids=self._output_indices,
                                              num_segments=self._units)  # [D, N]

        return tf.transpose(merged)  # [N, D]
