__all__ = ["plot_latent_space_with_cat", "plot_latent_space_with_con"]

from typing import Any

import matplotlib.cm as cm
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
        codes = np.unique(feature_values)
        for code in codes:
            category = feature_mapping[str(code)]
            is_category = (feature_values == code) & ~is_nan
            dims = np.take(latent_space.compress(is_category, axis=0), [0, 1], axis=1).T
            ax.scatter(*dims, label=category)
        dims = np.take(latent_space.compress(is_nan, axis=0), [0, 1], axis=1).T
        ax.scatter(*dims, label="NaN")
        ax.set(xlabel="dim 0", ylabel="dim 1")
        legend = ax.legend()
        legend.set_title(feature_name)
    return fig


def plot_latent_space_with_con(
    latent_space: FloatArray,
    feature_name: str,
    feature_values: FloatArray,
    style: str = DEFAULT_PLOT_STYLE,
    colormap: str = DEFAULT_DIVERGING_PALETTE,
) -> matplotlib.figure.Figure:
    """Plot a 2D latent space together with a colorbar mapping the latent
    space to the values of a continuous feature.

    Args:
        latent_space: Embedding, a ND array with at least two dimensions.
        feature_name: Name of continuous feature
        feature_values: Values of continuous feature
        style: Name of style to apply to the plot
        colormap: Name of colormap to use for the colorbar

    Raises:
        ValueError: If latent space does not have at least two dimensions.

    Returns:
        Figure
    """
    if latent_space.ndim < 2:
        raise ValueError("Expected at least two dimensions in latent space.")
    norm = TwoSlopeNorm(0.0, min(feature_values), max(feature_values))
    with style_settings(style):
        fig, ax = plt.subplots()
        dims = latent_space[:, 0], latent_space[:, 1]
        pts = ax.scatter(*dims, c=feature_values, cmap=colormap, norm=norm)
        cbar = fig.colorbar(pts, ax=ax)
        cbar.ax.set(ylabel=feature_name)
        ax.set(xlabel="dim 0", ylabel="dim 1")
    return fig


def plot_3D_latent_and_displacement(
    mu_baseline,
    mu_perturbed,
    feature_values,
    feature_name,
    show_baseline=True,
    show_perturbed=True,
    show_arrows=True,
    step: int = 1,
    altitude: int = 30,
    azimuth: int = 45,
):
    """
    Plot the movement of the samples in the 3D latent space after perturbing one
    input variable.

    Args:
        mu_baseline:
            ND array with dimensions n_samples x n_latent_nodes containing
            the latent representation of each sample
        mu_perturbed:
            ND array with dimensions n_samples x n_latent_nodes containing
            the latent representation of each sample after perturbing the input
        feature_values:
            1D array with feature values to map to a colormap ("bwr"). Each sample is
            colored according to its value for the feature of interest.
        feature_name:
            name of the feature mapped to a colormap
        show_baseline:
            plot orginal location of the samples in the latent space
        show_perturbed:
            plot final location (after perturbation) of the samples in latent space
        show_arrows:
            plot arrows from original to final location of each sample
        angle:
            elevation from dim1-dim2 plane for the visualization of latent space.

    Raises:
        ValueError: If latent space is not 3-dimensional (3 hidden nodes).
    Returns:
        Figure
    """
    if [np.shape(mu_baseline)[1], np.shape(mu_perturbed)[1]] != [3, 3]:
        raise ValueError(
            " The latent space must be 3-dimensional. Redefine num_latent to 3."
        )

    fig = plt.figure(layout="constrained", figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    ax.view_init(altitude, azimuth)

    if show_baseline:
        # vmin, vmax = np.min(feature_values[::step]), np.max(feature_values[::step])
        # abs_max = np.max([abs(vmin), abs(vmax)])
        ax.scatter(
            mu_baseline[::step, 0],
            mu_baseline[::step, 1],
            mu_baseline[::step, 2],
            marker=".",
            c=feature_values[::step],
            s=10,
            lw=0,
            cmap="seismic",
            vmin=-2,
            vmax=2,
        )
        ax.set_title(feature_name)
        fig.colorbar(
            cm.ScalarMappable(cmap="seismic", norm=Normalize(-2, 2)), ax=ax
        )  # Normalize(min(feature_values[::step]),max(feature_values[::step]))), ax=ax)
    if show_perturbed:
        ax.scatter(
            mu_perturbed[::step, 0],
            mu_perturbed[::step, 1],
            mu_perturbed[::step, 2],
            marker=".",
            color="lightblue",
            label="perturbed",
            lw=0.5,
        )
    if show_arrows:
        u = mu_perturbed[::step, 0] - mu_baseline[::step, 0]
        v = mu_perturbed[::step, 1] - mu_baseline[::step, 1]
        w = mu_perturbed[::step, 2] - mu_baseline[::step, 2]

        # module = np.sqrt(u * u + v * v + w * w)

        max_u, max_v, max_w = np.max(abs(u)), np.max(abs(v)), np.max(abs(w))
        # Arrow colors will be weighted contributions of
        # red -> dim1,
        # green -> dim2,
        # and blue-> dim3.
        # I.e. purple arrow means movement in dims 1 and 3
        colors = [
            (abs(du) / max_u, abs(dv) / max_v, abs(dw) / max_w, 0.7)
            for du, dv, dw in zip(u, v, w)
        ]
        ax.quiver(
            mu_baseline[::step, 0],
            mu_baseline[::step, 1],
            mu_baseline[::step, 2],
            u,
            v,
            w,
            color=colors,
            lw=0.8,
        )  # alpha=(1-module/np.max(module))**6, arrow_length_ratio=0)
        # help(ax.quiver)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")
    # ax.set_axis_off()

    return fig
