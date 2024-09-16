__all__ = ["axis_scale"]


import math
from typing import Literal, Sequence, Union

import numpy as np
import pandas as pd

from move.core.typing import FloatArray

Scale = Literal["log", "linear"]


def axis_scale(data: Union[FloatArray, pd.Series, Sequence[float]]) -> Scale:
    """Determine which scale (either log or linear) to use when plotting. If the data
    spans more than two orders of magnitude, log scale will be used."""
    ratio = math.log10(np.nanmax(data) / np.nanmin(data))
    if ratio > 2:
        return "log"
    return "linear"
