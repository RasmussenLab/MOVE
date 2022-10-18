__all__ = ["plot_categorical_feature_importance", "plot_continuous_feature_importance"]

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm

from move.core.typing import FloatArray
from move.visualization.style import (
    DEFAULT_DIVERGING_PALETTE,
    DEFAULT_QUALITATIVE_PALETTE,
    DEFAULT_PLOT_STYLE,
    color_cycle,
    style_settings,
)


def plot_categorical_feature_importance(
    diffs: FloatArray,
    feature_values: FloatArray,
    feature_names: list[str],
    feature_mapping: dict[str, int],
    style: str = DEFAULT_PLOT_STYLE,
    colormap: str = DEFAULT_QUALITATIVE_PALETTE,
) -> matplotlib.figure.Figure:
    """Plot a beeswarm displaying the top ten categorical features, based on
    their impact on the latent space when perturbed.

    Args:
        diffs:
            Impact caused by perturbing each feature, a 2D array (`num_samples`
            x `num_features`).
        feature_values:
            Values of the features, a 3D array (`num_samples` x `num_features`
            x `num_categories`).
        feature_names:
            Names of the features.
        feature_mapping:
            Mapping feature values to category names.
        style:
            Name of style to apply to the plot.
        colormap:
            Name of colormap to apply to the legend.

    Raises:
        ValueError:
            If inputs do not have expected number of dimensions or have
            ill-matched shapes.

    Returns:
        Figure
    """

    if feature_values.ndim != 3:
        raise ValueError("Expected feature values to have three dimensions.")
    if diffs.ndim != 2:
        raise ValueError("Expected differences to have two dimensions.")
    if feature_values[:, :, 0].shape != diffs.shape:
        raise ValueError("Feature values and differences shapes do not match.")

    # Select top 10 absolute sum difference
    top10_ids = np.argsort(np.sum(np.abs(diffs), axis=0))[::-1][:10]

    # Force figure aspect ratio to 1:1 or 2:1 (if less than 5 features)
    width: float = max(matplotlib.rcParams["figure.figsize"])
    figsize = (width, width)
    if top10_ids.size < 5:
        figsize = (width, width / 2)

    is_nan = (feature_values.sum(axis=2) == 0)[:, top10_ids].T.ravel()
    feature_values = np.argmax(feature_values, axis=2)  # 3D => 2D

    num_samples = diffs.shape[0]
    order = np.take(feature_names, top10_ids)
    perturbed_features = []
    for name in order:
        perturbed_features.extend([name] * num_samples)
    data = pd.DataFrame(
        dict(
            x=diffs.T[top10_ids, :].ravel()[~is_nan],
            y=np.compress(~is_nan, perturbed_features),
            category=feature_values.T[top10_ids, :].ravel()[~is_nan],
        )
    )
    with style_settings(style), color_cycle(colormap):
        fig, ax = plt.subplots(figsize=figsize)
        sns.swarmplot(data=data, x="x", y="y", hue="category", size=1, ax=ax)
        ax.set(xlabel="Impact on latent space", ylabel="Feature")
        # Fix labels in legend
        legend = ax.get_legend()
        assert legend is not None
        for text in legend.get_texts():
            code = text.get_text()
            if code in feature_mapping:
                text.set_text(feature_mapping[code])
    return fig


def plot_continuous_feature_importance(
    diffs: FloatArray,
    feature_values: FloatArray,
    feature_names: list[str],
    style: str = DEFAULT_PLOT_STYLE,
    colormap: str = DEFAULT_DIVERGING_PALETTE,
) -> matplotlib.figure.Figure:
    """Plot a beeswarm displaying the top ten continuous features, based on
    their impact on the latent space when perturbed.

    Args:
        diffs:
            Impact caused by perturbing each feature, a 2D array (`num_samples`
            x `num_features`).
        feature_values:
            Values of the features, a 2D array (`num_samples` x `num_features`).
        feature_names:
            Names of the features.
        style:
            Name of style to apply to the plot.
        colormap:
            Name of colormap to apply to the colorbar.

    Raises:
        ValueError:
            If inputs do not have two dimensions or have ill-matched shapes.

    Returns:
        Figure
    """
    if diffs.ndim != 2:
        raise ValueError("Expected differences to have two dimensions.")
    if feature_values[:, :].shape != diffs.shape:
        raise ValueError("Feature values and differences shapes do not match.")

    # Select top 10 absolute sum difference
    top10_ids = np.argsort(np.sum(np.abs(diffs), axis=0))[::-1][:10]

    # Force figure aspect ratio to 1:1 or 2:1 (if less than 5 features)
    width: float = max(matplotlib.rcParams["figure.figsize"])
    figsize = (width, width)
    if top10_ids.size < 5:
        figsize = (width, width / 2)

    is_nan = (feature_values == 0)[:, top10_ids].T.ravel()

    num_samples = diffs.shape[0]
    order = np.take(feature_names, top10_ids)
    perturbed_features = []
    for name in order:
        perturbed_features.extend([name] * num_samples)
    data = pd.DataFrame(
        dict(
            x=diffs.T[top10_ids, :].ravel()[~is_nan],
            y=np.compress(~is_nan, perturbed_features),
            value=feature_values.T[top10_ids, :].ravel()[~is_nan],
        )
    )

    # To obtain a colormap, we map the feature values to 25 discrete categories
    # using the two-slope norm. We then assign one color to each category
    # using the scalar mappable.
    vmin, vmax = data["value"].min(), data["value"].max()
    norm = TwoSlopeNorm(0.0, vmin, vmax)
    sm = ScalarMappable(norm, colormap)
    data["category"] = np.ma.compressed(norm(data["value"]) * 25).astype(int)
    palette = np.empty((25, 4))  # 25 colors x 4 channels
    palette[:13, :] = sm.to_rgba(np.linspace(vmin, 0, 13))  # first slope
    palette[12:, :] = sm.to_rgba(np.linspace(0, vmax, 13))  # second slope

    with style_settings(style):
        fig, ax = plt.subplots(figsize=figsize)
        sns.swarmplot(
            data=data, x="x", y="y", hue="category", ax=ax, palette=palette, size=2
        )
        ax.set(xlabel="Impact on latent space", ylabel="Feature")
        # Add colormap
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        cbar = fig.colorbar(sm, ax=ax)
        cbar.ax.set(ylabel="Feature value")
    return fig
