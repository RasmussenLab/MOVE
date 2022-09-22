__all__ = ["one_hot_encode", "scale"]

from typing import Optional

import numpy as np
from sklearn.preprocessing import scale as standardize


def one_hot_encode(x: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """One-hot encode a matrix with number of samples in its rows and number
    of features in its columns. Columns share number of classes.

    Args:
        x: a 1D or 2D matrix
        num_classes: expected number of classes. Automatically determined if
        None.

    Returns:
        3D one-hot encoded matrix, extra dim corresponds to number of classes
    """
    x = np.copy(x)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    # temporarily mark NaN as -1
    has_nan = np.any(np.isnan(x))
    x[np.isnan(x)] = np.nanmax(x) + 1
    x = x.astype(int)
    if num_classes is None:
        num_classes = x.max() + 1
    else:
        num_classes += has_nan
        if x.max() <= num_classes:
            raise ValueError(f"Expected {num_classes} classes, but found more.")
    encoded_x = np.zeros((x.size, num_classes), dtype=np.uint8)
    encoded_x[np.arange(x.size), x.ravel()] = 1
    encoded_x = encoded_x.reshape(*x.shape, num_classes)
    # remove NaN column
    if has_nan:
        encoded_x = encoded_x[:, :, :-1]
    # remove empty classes
    encoded_x = encoded_x[:, :, encoded_x.sum(axis=(0, 1)) > 0]
    return encoded_x


def scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center to mean and scale to unit variance. Convert NaN values to 0.

    Args:
        x: 2D array with samples in its rows and features in its columns

    Returns:
        Tuple containing (1) scaled output and (2) a 1D mask marking columns
        (i.e., features) with zero variance
    """
    logx = np.log2(x + 1)
    mask_1d = ~np.isclose(np.nanstd(logx, axis=0), 0.0)
    scaled_x = standardize(logx[:, mask_1d], axis=0)
    scaled_x[np.isnan(scaled_x)] = 0
    return scaled_x, mask_1d
