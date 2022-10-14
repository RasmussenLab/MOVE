__all__ = ["calculate_accuracy", "calculate_cosine_similarity"]

import numpy as np

from move.core.typing import FloatArray


def calculate_accuracy(
    original_input: FloatArray, reconstruction: FloatArray
) -> FloatArray:
    """Compute accuracy per sample.

    Args:
        original_input: Original labels (one-hot encoded as a 3D array).
        reconstruction: Reconstructed labels (2D array).

    Returns:
        Array of accuracy scores.
    """
    if original_input.ndim != 3:
        raise ValueError("Expected original input to have three dimensions.")
    if reconstruction.ndim != 2:
        raise ValueError("Expected reconstruction to have two dimensions.")
    if original_input[:, :, 0].shape != reconstruction.shape:
        raise ValueError("Original input and reconstruction shapes do not match.")

    is_nan = original_input.sum(axis=2) == 0
    original_input = np.argmax(original_input, axis=2)  # 3D => 2D
    y_true = np.ma.masked_array(original_input, mask=is_nan)
    y_pred = np.ma.masked_array(reconstruction, mask=is_nan)

    num_features = np.ma.count(y_true, axis=1)
    scores = np.ma.compressed(np.sum(y_true == y_pred, axis=1)) / num_features

    return scores


def calculate_cosine_similarity(
    original_input: FloatArray, reconstruction: FloatArray
) -> FloatArray:
    """Compute cosine similarity per sample.

    Args:
        original_input: Original values (2D array).
        reconstruction: Reconstructed values (2D array).

    Returns:
        Array of similarities.
    """
    if any((original_input.ndim != 2, reconstruction.ndim != 2)):
        raise ValueError("Expected both inputs to have two dimensions.")
    if original_input.shape != reconstruction.shape:
        raise ValueError("Original input and reconstruction shapes do not match.")

    is_nan = original_input == 0
    x = np.ma.masked_array(original_input, mask=is_nan)
    y = np.ma.masked_array(reconstruction, mask=is_nan)

    # Equivalent to `np.diag(sklearn.metrics.pairwise.cosine_similarity(x, y))`
    # But can handle masked arrays
    scores = np.ma.compressed(np.sum(x * y, axis=1)) / (norm(x) * norm(y))

    return scores


def norm(x: np.ma.MaskedArray, axis: int = 1) -> FloatArray:
    """Return Euclidean norm. This function is equivalent to `np.linalg.norm`,
    but it can handle masked arrays.

    Args:
        x: 2D masked array
        axis: Axis along which to the operation is performed. Defaults to 1.

    Returns:
        1D array with the specified axis removed.
    """
    return np.ma.compressed(np.sqrt(np.sum(x**2, axis=axis)))
