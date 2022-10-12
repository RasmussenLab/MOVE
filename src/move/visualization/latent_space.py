__all__ = [
    "plot_latent_space_with_cat",
    "plot_latent_space_with_con"
]

import numpy as np
import matplotlib.figure
import matplotlib.style
import matplotlib.pyplot as plt

from move.core.typing import BoolArray, FloatArray
from move.visualization.style import style_settings


def plot_latent_space_with_cat(
    latent_space: FloatArray,
    feature_name: str,
    feature_values: FloatArray,
    feature_mapping: dict[str, int],
    is_nan_mask: BoolArray
) -> matplotlib.figure.Figure:
    """Plots a 2D latent space together with a legend mapping the latent
    space to the values of a discrete feature.

    Args:
        latent_space:
            Embedding, a ND array with at least two dimensions.
        feature_name:
            Name of categorical feature
        feature_values:
            Values of categorical feature
        feature_mapping:
            Mapping of codes to categories for the categorical feature
        is_nan_mask:
            Array of bool values indicating which feature values are NaNs

    Raises:
        ValueError: If latent space does not have at least two dimensions.

    Returns:
        Figure
    """
    if latent_space.ndim < 2:
        raise ValueError("Expected at least two dimensions in latent space.")
    with style_settings("ggplot"):
        fig, ax = plt.subplots()
        codes = np.unique(feature_values)
        for code in codes:
            category = feature_mapping[str(code)]
            is_category = (feature_values == code) & ~is_nan_mask
            dims = latent_space[is_category, [0, 1]].T
            ax.scatter(*dims, label=category)
        dims = latent_space[is_nan_mask, [0, 1]].T
        ax.scatter(*dims, label="NaN")
        legend = ax.legend()
        legend.set_title(feature_name)
    return fig


def plot_latent_space_with_con(
    latent_space: FloatArray,
    feature_name: str,
    feature_values: FloatArray,
) -> matplotlib.figure.Figure:
    """Plots a 2D latent space together with a colorbar mapping the latent
    space to the values of a continuous feature.

    Args:
        latent_space: Embedding, a ND array with at least two dimensions.
        feature_name: Name of continuous feature
        feature_values: Values of continuous feature

    Raises:
        ValueError: If latent space does not have at least two dimensions.

    Returns:
        Figure
    """
    if latent_space.ndim < 2:
        raise ValueError("Expected at least two dimensions in latent space.")
    with style_settings("ggplot"):
        fig, ax = plt.subplots()
        dims = latent_space[:, 0], latent_space[:, 1]
        pts = ax.scatter(*dims, c=feature_values)
        cbar = fig.colorbar(pts, ax=ax)
        cbar.ax.set(ylabel=feature_name)
        ax.set(xlabel="dim 0", ylabel="dim 1")
    return fig
