__all__ = ["plot_value_distributions"]

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np

from move.core.typing import FloatArray
from move.visualization.style import (
    DEFAULT_PLOT_STYLE,
    style_settings,
)

def plot_value_distributions(
    feature_values: FloatArray,
    feature_names: list[str],
    style: str = DEFAULT_PLOT_STYLE,
    nbins: int = 50,
) -> matplotlib.figure.Figure:
    """
    Given a certain dataset and its feature labels,
    plot the distribution of values.
    
    Args:
        feature_values:
            Values of the features, a 2D array (`num_samples` x `num_features`).
        feature_names:
            Names of the features.
        style:
            Name of style to apply to the plot.
        colormap:
            Name of colormap to apply to the colorbar.

    Returns:
        Figure
    """
    width: float = max(matplotlib.rcParams["figure.figsize"])
    figsize = (width, width)
    with style_settings(style):
        fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=figsize)
        for i, feature in enumerate(feature_names): 
            feat_i_list = feature_values[:,i]
            vmin, vmax = np.nanmin(feature_values[:,i]) ,  np.nanmax(feature_values[:,i])
            ax.hist(feat_i_list, bins = np.linspace(vmin,vmax,nbins), label=feature, histtype='step')    
            ax.set_xlabel("Feature value")
            ax.set_ylabel("Frequency")
        #ax.legend()
    return fig


