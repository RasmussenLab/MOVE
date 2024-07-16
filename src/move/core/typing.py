__all__ = [
    "BoolArray",
    "FloatArray",
    "IntArray",
    "ObjectArray",
    "EncodedData",
    "PathLike",
]

import os
from typing import Literal, TypedDict, Union

import numpy as np
import torch
from numpy.typing import NDArray

LoggingLevel = Union[
    int,
    Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
]
PathLike = Union[str, os.PathLike]

Split = Literal["all", "train", "valid", "test"]


class IndicesDict(TypedDict):
    train_indices: torch.Tensor
    test_indices: torch.Tensor
    valid_indices: torch.Tensor


BoolArray = NDArray[np.bool_]
IntArray = Union[NDArray[np.int_], NDArray[np.uint]]
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
