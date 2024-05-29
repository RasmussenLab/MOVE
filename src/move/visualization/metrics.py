__all__ = ["plot_metrics_boxplot"]

from collections.abc import Sequence
from typing import Union

import pandas as pd
import matplotlib
import matplotlib.figure

from move.core.typing import FloatArray
from move.visualization.figure import create_figure
from move.visualization.style import (
    DEFAULT_PLOT_STYLE,
    DEFAULT_QUALITATIVE_PALETTE,
    color_cycle,
    style_settings,
)


def plot_metrics_boxplot(
    scores: Union[Sequence[FloatArray], pd.DataFrame],
    labels: Sequence[str],
    style: str = DEFAULT_PLOT_STYLE,
    colormap: str = DEFAULT_QUALITATIVE_PALETTE,
) -> matplotlib.figure.Figure:
    """Plot a box plot, showing the distribution of metrics per dataset. Each
    score corresponds (for example) to a sample.

    Args:
        scores: List of dataset metrics or DataFrame
        labels: List of dataset names
        style: Name of style to apply to the plot
        colormap: Name of colormap to use for the boxes

    Returns:
        Figure
    """
    values = scores.values if isinstance(scores, pd.DataFrame) else scores
    with style_settings(style), color_cycle(colormap):
        labelcolor = matplotlib.rcParams["axes.labelcolor"]
        fig, ax = create_figure()
        comps = ax.boxplot(
            values,
            labels=labels,
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
        ax.invert_yaxis()
    return fig
