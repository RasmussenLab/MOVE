__all__ = ["create_figure", "show_figure"]

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def create_figure(**fig_kw) -> tuple[Figure, Axes]:
    """Create a figure.

    Returns:
        A tuple containing a Figure and an Axes object. Unlike the customary
        (and equivalent) `matplotlib.pyplot.subplots` function, this method is
        correctly typed. That's the only difference."""
    fig, ax = plt.subplots(**fig_kw)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    return fig, ax


def show_figure(fig: Figure) -> None:
    """Display a figure if running inside an interactive IPython/Jupyter
    session (e.g., a tutorial notebook). """
    try:
        from IPython import get_ipython
    except ImportError:
        return
    if get_ipython() is not None:
        plt.show()
