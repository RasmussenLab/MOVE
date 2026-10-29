__all__ = ["plot_latent_space_with_cat", "plot_latent_space_with_con"]

from typing import Any, Optional, cast

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm

from move.core.typing import BoolArray, FloatArray
from move.visualization.style import (
    DEFAULT_DIVERGING_PALETTE,
    DEFAULT_PLOT_STYLE,
    DEFAULT_QUALITATIVE_PALETTE,
    color_cycle,
    style_settings,
)


def plot_latent_space_with_cat(
    latent_space: FloatArray,
    feature_name: str,
    feature_values: FloatArray,
    feature_mapping: dict[str, Any],
    is_nan: BoolArray,
    style: str = DEFAULT_PLOT_STYLE,
    colormap: str = DEFAULT_QUALITATIVE_PALETTE,
) -> matplotlib.figure.Figure:
    """Plot a 2D latent space together with a legend mapping the latent
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
        is_nan:
            Array of bool values indicating which feature values are NaNs
        style:
            Name of style to apply to the plot
        colormap:
            Name of qualitative colormap to use for each category

    Raises:
        ValueError: If latent space does not have at least two dimensions.

    Returns:
        Figure
    """
    if latent_space.ndim < 2:
        raise ValueError("Expected at least two dimensions in latent space.")
    with style_settings(style), color_cycle(colormap):
        fig, ax = plt.subplots()
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        codes = np.unique(feature_values)
        for code in codes:
            category = feature_mapping[str(code)]
            is_category = (feature_values == code) & ~is_nan
            dims = np.take(latent_space.compress(is_category, axis=0), (0, 1), axis=1).T
            ax.scatter(*dims, label=category)
        dims = np.take(latent_space.compress(is_nan, axis=0), (0, 1), axis=1).T
        ax.scatter(*dims, label="NaN")
        ax.set(xlabel="dim 0", ylabel="dim 1")
        legend = ax.legend()
        legend.set_title(feature_name)
    return fig


def plot_latent_space_with_con(
    latent_space: FloatArray,
    feature_name: str,
    feature_values: FloatArray,
    is_nan: Optional[BoolArray] = None,
    style: str = DEFAULT_PLOT_STYLE,
    colormap: str = DEFAULT_DIVERGING_PALETTE,
) -> matplotlib.figure.Figure:
    """Plot a 2D latent space together with a colorbar mapping the latent
    space to the values of a continuous feature. Samples with a missing
    feature value are drawn in dark gray.

    Args:
        latent_space: Embedding, a ND array with at least two dimensions.
        feature_name: Name of continuous feature
        feature_values: Values of continuous feature
        is_nan: Array of bool values indicating which feature values are NaNs
        style: Name of style to apply to the plot
        colormap: Name of colormap to use for the colorbar

    Raises:
        ValueError: If latent space does not have at least two dimensions.

    Returns:
        Figure
    """
    if latent_space.ndim < 2:
        raise ValueError("Expected at least two dimensions in latent space.")
    feature_values = np.ravel(feature_values)
    is_nan = np.isnan(feature_values) if is_nan is None else np.ravel(is_nan)
    observed = feature_values[~is_nan]
    vmin, vmax = (observed.min(), observed.max()) if observed.size else (-1.0, 1.0)
    norm = (
        TwoSlopeNorm(0.0, vmin, vmax)
        if vmin < 0.0 < vmax
        else Normalize(vmin, vmax)
    )
    with style_settings(style):
        fig, ax = plt.subplots()
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        if is_nan.any():
            dims = latent_space[is_nan, 0], latent_space[is_nan, 1]
            ax.scatter(*dims, c="dimgray", label="NaN")
            ax.legend()
        dims = latent_space[~is_nan, 0], latent_space[~is_nan, 1]
        pts = ax.scatter(*dims, c=cast(list, observed), cmap=colormap, norm=norm)
        cbar = fig.colorbar(pts, ax=ax)
        cbar.ax.set(ylabel=feature_name)
        ax.set(xlabel="dim 0", ylabel="dim 1")
    return fig
