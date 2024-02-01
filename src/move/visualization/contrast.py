__all__ = ["get_luminance", "get_contrast_ratio", "MIN_CONTRAST"]

from typing import Sequence, Union

import numpy as np
from numpy.typing import NDArray

MIN_CONTRAST = 4.5
# https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html

Color = Union[Sequence[float], NDArray]


def get_luminance(color: Color) -> float:
    """Compute relative brightness of any point in a colorspace, normalized
    to 0 for darkest black and 1 for lightest white.

    Args:
        color:
            Array or three-element tuple representing a color in terms of red,
            green, and blue coordinates
    """
    # http://www.w3.org/TR/2008/REC-WCAG20-20081211/#relativeluminancedef
    color = np.asarray(color)
    w = np.array([0.2126, 0.7152, 0.0722])
    rgb = np.piecewise(
        color,
        [color <= 0.03928, color > 0.03928],
        [lambda x: x / 12.92, lambda x: ((x + 0.055) / 1.055) ** 2.4],
    )
    return (w @ rgb).item()


def get_contrast_ratio(color1: Color, color2: Color) -> float:
    """Compute ratio between the relative luminance of the lighter and the
    darker of two colors. Contrast ratios range from 1 to 21.

    Args:
        color1:
        color2:
            Array or three-element tuple representing a color in terms of red,
            green, and blue coordinates
    """
    # http://www.w3.org/TR/2008/REC-WCAG20-20081211/#contrast-ratiodef
    lum1 = get_luminance(color1)
    lum2 = get_luminance(color2)
    ratio = (lum1 + 0.05) / (lum2 + 0.05)
    if lum2 > lum1:
        return ratio**-1
    return ratio
