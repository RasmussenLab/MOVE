__all__ = ["plot_loss_curves", "LOSS_LABELS"]

from collections.abc import Sequence

import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt

from move.visualization.style import color_cycle, style_settings


LOSS_LABELS = ("Loss", "Cross-Entropy", "Sum of Squared Errors", "KLD")


def plot_loss_curves(
    losses: Sequence[list[float]],
    labels: Sequence[str] = LOSS_LABELS,
    style: str = "ggplot",
    colormap: str = "Dark2",
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
