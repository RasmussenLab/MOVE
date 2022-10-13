__all__ = ["plot_loss_curves", "LOSS_LABELS"]

import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt

from move.visualization.style import color_cycle, style_settings


LOSS_LABELS = ("Loss", "Cross-Entropy", "Sum of Squared Errors", "KLD")


def plot_loss_curves(
    losses: tuple[list[float], ...],
    labels: tuple[str, ...] = LOSS_LABELS,
    style: str = "ggplot",
    colormap: str = "Dark2",
) -> matplotlib.figure.Figure:
    """Plots one or more loss curves in ggplot style.

    Args:
        losses: Tuple containing lists of loss values
        labels: Tuple containing names of each loss line

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
