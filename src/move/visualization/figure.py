__all__ = ["create_figure"]

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def create_figure() -> tuple[Figure, Axes]:
    """Create a figure.

    Returns:
        A tuple containing a Figure and an Axes object. Unlike the customary
        (and equivalent) `matplotlib.pyplot.subplots` function, this method is
        correctly typed. That's the only difference."""
    fig, ax = plt.subplots()
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    return fig, ax
