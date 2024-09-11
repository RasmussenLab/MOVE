__all__ = ["plot_metrics_boxplot"]

from collections.abc import Sequence
from typing import Callable, Optional, Union, cast

import matplotlib
import matplotlib.figure
import pandas as pd

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
    labels: Optional[Sequence[str]],
    style: str = DEFAULT_PLOT_STYLE,
    colormap: str = DEFAULT_QUALITATIVE_PALETTE,
) -> matplotlib.figure.Figure:
    """Plot a box plot, showing the distribution of metrics per dataset. Each
    score corresponds (for example) to a sample.

    Args:
        scores: List of dataset metrics or DataFrame
        labels: List of dataset names. If None, DataFrame column names will be used.
        style: Name of style to apply to the plot
        colormap: Name of colormap to use for the boxes

    Returns:
        Figure
    """
    is_df = isinstance(scores, pd.DataFrame)
    not_na: Callable[[pd.Series], pd.Series] = lambda sr: sr.notna()
    values = [scores[col][not_na].values for col in scores.columns] if is_df else scores  # type: ignore
    if labels is None:
        if not is_df:
            raise ValueError("Label names missing")
        labels = cast(Sequence[str], scores.columns)
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
