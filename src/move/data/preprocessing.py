__all__ = ["one_hot_encode", "one_hot_encode_single", "scale"]

from typing import Any, Optional

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.preprocessing import scale as standardize

from move.core.typing import BoolArray, FloatArray, IntArray


def _category_name(value: Any) -> str:
    return value if isinstance(value, str) else str(int(value))


def one_hot_encode(x_: ArrayLike) -> tuple[IntArray, dict[str, int]]:
    """One-hot encode a matrix with samples in its rows and features in its
    columns. Columns share number of classes.

    Args:
        x: a 1D or 2D matrix, can be numerical or contain strings

    Returns:
        A 3D one-hot encoded matrix (extra dim corresponds to number of
        classes) and a mapping between classes and corresponding codes
    """
    x: np.ndarray = np.copy(x_)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    shape = x.shape
    has_na = np.any(pd.isna(x))
    if x.dtype == object:
        x = x.astype(str)
    categories, codes = np.unique(x, return_inverse=True)
    num_classes = len(categories)
    encoded_x = np.zeros((x.size, num_classes), dtype=np.uint8)
    encoded_x[np.arange(x.size), codes.astype(np.uint8).ravel()] = 1
    encoded_x = encoded_x.reshape(*shape, num_classes)
    if has_na:
        # remove NaN column
        categories = categories[:-1]
        encoded_x = encoded_x[:, :, :-1]
    mapping = {
        _category_name(category): code for code, category in enumerate(categories)
    }
    return encoded_x, mapping


def one_hot_encode_single(mapping: dict[str, int], value: Optional[str]) -> IntArray:
    """One-hot encode a single value given an existing mapping.

    Args:
        mapping: cateogry-to-code lookup dictionary
        value: category

    Returns:
        2D array
    """
    encoded_value = np.zeros((1, len(mapping)))
    if not pd.isna(value):
        code = mapping[str(value)]
        encoded_value[0, code] = 1
    return encoded_value


def scale(x: np.ndarray) -> tuple[FloatArray, BoolArray]:
    """Center to mean and scale to unit variance. Convert NaN values to 0.

    Args:
        x: 2D array with samples in its rows and features in its columns

    Returns:
        Tuple containing (1) scaled output and (2) a 1D mask marking columns
        (i.e., features) without zero variance
    """
    logx = np.log2(x + 1)
    mask_1d = ~np.isclose(np.nanstd(logx, axis=0), 0.0)
    scaled_x = standardize(logx[:, mask_1d], axis=0)
    scaled_x[np.isnan(scaled_x)] = 0
    return scaled_x, mask_1d
