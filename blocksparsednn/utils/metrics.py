import numpy as np


def count_correct(logits: np.ndarray, labels: np.ndarray) -> int:
    """
    Computes the number of correct predictions.

    Args:
        logits: A [B, K] array of log probabilities for each batch sample
        labels: A [B] array of integer labels for each batch sample
    """
    assert logits.shape[0] == labels.shape[0], 'Misaligned arrays: {0}, {1}'.format(logits.shape, labels.shape)
    pred = np.argmax(logits, axis=-1).astype(int)  # [B]
    return np.sum(np.equal(pred, labels).astype(int))  # Scalar


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    correct = count_correct(logits=logits, labels=labels)
    return correct / logits.shape[0]
