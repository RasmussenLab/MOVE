__all__ = ["plot_metrics_boxplot"]

from collections.abc import Sequence

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt

from move.core.typing import FloatArray
from move.visualization.style import (
    DEFAULT_QUALITATIVE_PALETTE,
    DEFAULT_PLOT_STYLE,
    color_cycle,
    style_settings,
)


def plot_metrics_boxplot(
    scores: Sequence[FloatArray],
    labels: Sequence[str],
    style: str = DEFAULT_PLOT_STYLE,
    colormap: str = DEFAULT_QUALITATIVE_PALETTE,
) -> matplotlib.figure.Figure:
    """Plot a box plot, showing the distribution of metrics per dataset. Each
    score corresponds (for example) to a sample.

    Args:
        scores: List of dataset metrics
        labels: List of dataset names
        style: Name of style to apply to the plot
        colormap: Name of colormap to use for the boxes

    Returns:
        Figure
    """
    with style_settings(style), color_cycle(colormap):
        labelcolor = matplotlib.rcParams["axes.labelcolor"]
        fig, ax = plt.subplots()
        comps = ax.boxplot(
            scores[::-1],
            labels=labels[::-1],
            patch_artist=True,
            vert=False,
            capprops=dict(color=labelcolor),
            flierprops=dict(
                marker="d",
                markersize=5,
                markerfacecolor=labelcolor,
                markeredgecolor=labelcolor,
            ),
            medianprops=dict(color=labelcolor),
            whiskerprops=dict(color=labelcolor),
        )
        prop_cycle = matplotlib.rcParams["axes.prop_cycle"]
        for box, prop in zip(comps["boxes"], prop_cycle()):
            box.update(dict(facecolor=prop["color"], edgecolor=labelcolor))
        ax.set(xlim=(-0.05, 1.05), xlabel="Score", ylabel="Dataset")
    return fig
