__all__ = [
    "DEFAULT_DIVERGING_PALETTE",
    "DEFAULT_QUALITATIVE_PALETTE",
    "DEFAULT_PLOT_STYLE",
    "color_cycle",
    "style_settings",
]

from typing import ContextManager, cast

import matplotlib
import matplotlib.style
from cycler import cycler
from matplotlib.cm import ColormapRegistry
from matplotlib.colors import ListedColormap

DEFAULT_DIVERGING_PALETTE = "RdYlBu"
DEFAULT_QUALITATIVE_PALETTE = "Dark2"
DEFAULT_PLOT_STYLE = "ggplot"


def color_cycle(colormap: str) -> ContextManager:
    """Returns a context manager for using a color cycle in plots.

    Args:
        colormap: Name of qualitative color map.

    Returns:
        Context manager
    """
    registry: ColormapRegistry = matplotlib.colormaps
    colormap_ = registry[colormap]
    if isinstance(colormap_, ListedColormap):
        prop_cycle = cycler(color=colormap_.colors)
        return matplotlib.rc_context({"axes.prop_cycle": prop_cycle})
    raise ValueError("Only colormaps that are list of colors supported.")


def style_settings(style: str) -> ContextManager:
    """Returns a context manager for setting a plot's style.

    Args:
        style: Style name.

    Returns:
        Context manager
    """
    return cast(ContextManager, matplotlib.style.context(style))
