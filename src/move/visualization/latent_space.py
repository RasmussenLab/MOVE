
import numpy as np
import matplotlib.figure
import matplotlib.style
import matplotlib.pyplot as plt

from move.core.typing import BoolArray, FloatArray
from move.visualization.style import style_settings


def plot_latent_space_with_continuous(
    latent_space: FloatArray,
    feature_name: str,
    feature_values: FloatArray,
    **scatter_args
) -> matplotlib.figure.Figure:
    if latent_space.ndim < 2:
        raise ValueError("Expected at least 2 dimensions in latent space.")
    with style_settings("ggplot"):
        fig, ax = plt.subplots()
        dims = latent_space[:, 0], latent_space[:, 1]
        pts = ax.scatter(*dims, c=feature_values, **scatter_args)
        cbar = fig.colorbar(pts, ax=ax)
        cbar.ax.set(ylabel=feature_name)
        ax.set(xlabel="dim 0", ylabel="dim 1")
    return fig
