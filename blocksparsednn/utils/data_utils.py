import numpy as np

def get_seq_length(embeddings: np.ndarray) -> np.ndarray:
    """
    Returns the actual sequence length in the padded embeddings.

    Args:
        embeddings: A [B, T, D] array of feature vectors (D) for each sample (B) and seq element (T)
    Returns:
        A [B] tensor containing the sequence lengths for each sample.
    """
    nonzero_mask = np.any(np.logical_not(np.isclose(embeddings, 0.0)), axis=-1).astype(int)  # [B, T], 0 when vector is all zeros, 1 otherwise
    seq_idx = np.expand_dims(np.arange(embeddings.shape[1]), axis=0)  # [1, T]

    masked_idx = nonzero_mask * seq_idx  # [B, T]
    return np.max(masked_idx, axis=-1) + 1  # [B]
