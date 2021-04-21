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

        self._weights: List[tf.Variable] = []
        self._output_indices: List[int] = []

        input_indices = []
        for r in range(self._pattern.shape[0]):
            for c in range(self._pattern.shape[1]):
                if self._pattern[r, c] == 1:
                    # Create the weight variable
                    w = tf.compat.v1.get_variable('w-{0}-{1}'.format(r, c),
                                                  shape=[block_size, block_size],
                                                  dtype=dtype,
                                                  initializer=initializer)
                    self._weights.append(w)

                    # Add the feature index
                    input_indices.append(r*block_size)

                    # Add the output indices
                    self._output_indices.extend((c * block_size + i) for i in range(block_size))

        self._input_indices: tf.Variable = tf.Variable(input_indices)
        self._blocks: tf.Tensor = tf.stack(values=self._weights)
    
    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Executes the block sparse matrix multiplication on the given inputs.

        Args:
            inputs: A [N, D] tensor containing features (D) for each batch element (N)
        Returns:
            A [N, D] tensor after applying the block sparse matrix.
        """

        def body(i, mul_result_list):
            start_idx = self._input_indices[i]
            w = self._blocks[start_idx] # [B, B]
            input_slice = tf.slice(inputs, begin=[0, start_idx], size=[-1, self._block_size])  # [N, B]
            mul = tf.linalg.matmul(input_slice, w) # [N, B]
            mul_result_list = mul_result_list.write(i, mul)
            return [i+1, mul_result_list]

        def cond(i, mul_result_list):
            return tf.less(i, self._nnz)

        i = 0
        mul_result_list: tf.TensorArray = tf.TensorArray(self._dtype,
                size=self._nnz, clear_after_read=False)
        _, result = tf.while_loop(cond, body, [i, mul_result_list], parallel_iterations=self._nnz)

        mul_results = [result.read(i) for i in range(self._nnz) ]
        mul_results = tf.concat(mul_results, axis=-1) # [N, B * NNZ]

        merged = tf.math.unsorted_segment_sum(tf.transpose(mul_results),
                                              segment_ids=self._output_indices,
                                              num_segments=self._units)  # [D, N]

        return tf.transpose(merged)  # [N, D]
