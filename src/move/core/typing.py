__all__ = ["BoolArray", "FloatArray", "IntArray", "ObjectArray", "PathLike"]

import os
from typing import Union

import numpy as np
from numpy.typing import NDArray

PathLike = Union[str, os.PathLike]

BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float_]
ObjectArray = NDArray[np.object_]
