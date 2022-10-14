__all__ = ["calculate_accuracy", "calculate_cosine_similarity"]

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

from move.core.typing import IntArray, FloatArray


def calculate_accuracy(original_input: IntArray, reconstruction: IntArray) -> float:
    """Computes accuracy.

    Args:
        original_input: Original labels (one-hot encoded as a 3D array)
        reconstruction: Reconstructed labels (2D array)

    Returns:
        Fraction of correctly reconstructed samples
    """
    if original_input.ndim != 3:
        raise ValueError("Expected original input to have three dimensions.")
    if reconstruction.ndim != 2:
        raise ValueError("Expected reconstruction to have two dimensions.")
    not_nan_ids = np.flatnonzero(original_input.sum(axis=2) > 0)
    original_input = np.argmax(original_input, axis=2)  # 3D => 2D
    y_true = np.take(original_input, not_nan_ids)
    y_pred = np.take(reconstruction, not_nan_ids)
    return accuracy_score(y_true, y_pred)


def calculate_cosine_similarity(
    original_input: FloatArray, reconstruction: FloatArray
) -> float:
    """Computes cosine similarity.

    Args:
        original_input: Original values (2D array)
        reconstruction: Reconstructed values (2D array)

    Returns:
        Similarity between original and reconstructed values
    """
    if any((original_input.ndim != 2, reconstruction.ndim != 2)):
        raise ValueError("Expected both inputs to have two dimensions.")
    not_nan_ids = np.flatnonzero(original_input != 0)
    x = np.expand_dims(np.take(original_input, not_nan_ids), axis=0)
    y = np.expand_dims(np.take(reconstruction, not_nan_ids), axis=0)
    return cosine_similarity(x, y).item()
