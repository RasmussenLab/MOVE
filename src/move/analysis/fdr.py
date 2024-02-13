import math
from typing import cast

import numpy as np
from numpy.typing import NDArray


def argnearest(array: NDArray, target: float) -> int:
    """Find value in array closest to target. Assumes array is sorted in
    ascending order."""
    idx = np.searchsorted(array, target, side="left")
    if idx > 0 and (
        idx == len(array)
        or math.fabs(target - array[idx - 1]) < math.fabs(target - array[idx])
    ):
        return cast(int, idx - 1)
    else:
        return cast(int, idx)
