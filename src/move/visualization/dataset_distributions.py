__all__ = ["plot_value_distributions"]

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from move.core.typing import FloatArray
from move.visualization.style import DEFAULT_PLOT_STYLE, style_settings


def plot_value_distributions(
    feature_values: FloatArray,
    style: str = "fast",
    nbins: int = 100,
) -> matplotlib.figure.Figure:
    """
    Given a certain dataset, plot its distribution of values.


    Args:
        feature_values:
            Values of the features, a 2D array (`num_samples` x `num_features`).
        style:
            Name of style to apply to the plot.
        colormap:
            Name of colormap to apply to the colorbar.

    Returns:
        Figure
    """
    vmin, vmax = np.nanmin(feature_values), np.nanmax(feature_values)
    with style_settings(style):
        fig = plt.figure(layout="constrained")
        ax = fig.add_subplot(projection="3d")
        x_val = np.linspace(vmin, vmax, nbins)
        y_val = np.arange(np.shape(feature_values)[1])
        x_val, y_val = np.meshgrid(x_val, y_val)

        histogram = []
        for i in range(np.shape(feature_values)[1]):
            feat_i_list = feature_values[:, i]
            feat_hist, feat_bin_edges = np.histogram(
                feat_i_list, bins=nbins, range=(vmin, vmax)
            )
            histogram.append(feat_hist)

        ax.plot_surface(x_val, y_val, np.array(histogram), cmap="viridis")
        ax.set_xlabel("Feature value")
        ax.set_ylabel("Feature ID number")
        ax.set_zlabel("Frequency")
        # ax.legend()
    return fig
