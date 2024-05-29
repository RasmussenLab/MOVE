__all__ = ["LOSS_LABELS", "plot_loss_curves"]

from collections.abc import Sequence
from typing import Union

import matplotlib.figure
import numpy as np
import pandas as pd

from move.visualization.figure import create_figure
from move.visualization.style import (
    DEFAULT_PLOT_STYLE,
    DEFAULT_QUALITATIVE_PALETTE,
    color_cycle,
    style_settings,
)

LOSS_LABELS = (
    "Loss",
    "Reconstruction error (discrete)",
    "Reconstruction error (continuous)",
    "Regularization error",
)


def plot_loss_curves(
    losses: Union[Sequence[list[float]], pd.DataFrame],
    labels: Sequence[str] = LOSS_LABELS,
    style: str = DEFAULT_PLOT_STYLE,
    colormap: str = DEFAULT_QUALITATIVE_PALETTE,
    xlabel: str = "Epochs",
) -> matplotlib.figure.Figure:
    """Plot one or more loss curves.

    Args:
        losses: List containing lists of loss values or a DataFrame
        labels: List containing names of each loss line
        style: Name of style to apply to the plot
        colormap: Name of colormap to use for the curves

    Returns:
        Figure
    """
    is_df = isinstance(losses, pd.DataFrame)
    num_epochs = len(losses[0])
    epochs = np.arange(num_epochs)
    with style_settings(style), color_cycle(colormap):
        fig, ax = create_figure()
        for i, label in enumerate(labels):
            if is_df:
                colname = losses.columns[i]
                loss = losses[colname]
            else:
                loss = losses[i]
            ax.plot(epochs, loss, label=label, linestyle="-")
        ax.legend()
        ax.set(xlabel=xlabel, ylabel="Loss")
    return fig
