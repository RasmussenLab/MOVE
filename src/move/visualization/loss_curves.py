__all__ = ["LOSS_LABELS", "plot_loss_curves"]

from collections.abc import Sequence

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from move.visualization.style import (
    DEFAULT_QUALITATIVE_PALETTE,
    DEFAULT_PLOT_STYLE,
    color_cycle,
    style_settings,
)

LOSS_LABELS = ("Loss", "Cross-Entropy", "Sum of Squared Errors", "KLD")


def plot_loss_curves(
    losses: Sequence[list[float]],
    labels: Sequence[str] = LOSS_LABELS,
    style: str = DEFAULT_PLOT_STYLE,
    colormap: str = DEFAULT_QUALITATIVE_PALETTE,
) -> matplotlib.figure.Figure:
    """Plot one or more loss curves.

    Args:
        losses: List containing lists of loss values
        labels: List containing names of each loss line
        style: Name of style to apply to the plot
        colormap: Name of colormap to use for the curves

    Returns:
        Figure
    """
    num_epochs = len(losses[0])
    epochs = np.arange(num_epochs)
    with style_settings(style), color_cycle(colormap):
        fig, ax = plt.subplots()
        for loss, label in zip(losses, labels):
            ax.plot(epochs, loss, label=label, linestyle="-")
        ax.legend()
        ax.set(xlabel="Epochs", ylabel="Loss")
    return fig
