__all__ = [
    "BoolArray",
    "FloatArray",
    "IntArray",
    "ObjectArray",
    "EncodedData",
    "PathLike",
]

import os
from typing import TypedDict, Union

import numpy as np
import torch
from numpy.typing import NDArray

PathLike = Union[str, os.PathLike]

BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float_]
ObjectArray = NDArray[np.object_]


class EncodedData(TypedDict):
    """Dictionary containing a tensor, a name, and a list of feature names."""

    dataset_name: str
    tensor: torch.Tensor
    feature_names: list[str]


class EncodedDiscreteData(EncodedData):
    """Dictionary containing a tensor, a name, a list of feature names, and a
    mapping."""

    mapping: dict[str, int]
