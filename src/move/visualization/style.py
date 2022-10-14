__all__ = ["color_cycle", "style_settings"]

from typing import cast, ContextManager

import matplotlib
import matplotlib.style
from cycler import cycler
from matplotlib.cm import ColormapRegistry
from matplotlib.colors import ListedColormap


def color_cycle(colormap: str) -> ContextManager:
    """Returns a context manager for using a color cycle in plots.

    Args:
        colormap: Name of qualitative color map.

    Returns:
        Context manager
    """
    registry: ColormapRegistry = getattr(matplotlib, "colormaps")
    colormap = registry[colormap]
    if not isinstance(colormap, ListedColormap):
        raise ValueError("Only colormaps that are list of colors supported.")
    prop_cycle = cycler(color=getattr(colormap, "colors"))
    return matplotlib.rc_context({"axes.prop_cycle": prop_cycle})


def style_settings(style: str) -> ContextManager:
    """Returns a context manager for setting a plot's style.

    Args:
        style: Style name.

    Returns:
        Context manager
    """
    return cast(ContextManager, matplotlib.style.context(style))