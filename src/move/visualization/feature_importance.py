__all__ = ["plot_categorical_feature_importance"]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from move.core.typing import FloatArray
from move.visualization.style import color_cycle, style_settings


def plot_categorical_feature_importance(
    diffs: FloatArray,
    feature_values: FloatArray,
    feature_names: list[str],
    feature_mapping: dict[str, int],
    style: str = "ggplot",
    colormap: str = "Dark2",
):
    if feature_values.ndim != 3:
        raise ValueError("Expected feature values to have three dimensions.")
    if diffs.ndim != 2:
        raise ValueError("Expected differences to have two dimensions.")
    if feature_values[:, :, 0].shape != diffs.shape:
        raise ValueError("Feature values and differences shapes do not match.")

    # Select top 10 absolute sum difference
    top10_ids = np.argsort(np.sum(np.abs(diffs), axis=0))[::-1][:10]

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
        fig, ax = plt.subplots()
        sns.swarmplot(
            data=data, x="x", y="y", hue="category", order=order, size=1, ax=ax
        )
        ax.set(
            xlabel="Impact on latent space",
            ylabel="Feature",
        )
        # Fix labels in legend
        legend = ax.get_legend()
        assert legend is not None
        for text in legend.get_texts():
            code = text.get_text()
            if code in feature_mapping:
                text.set_text(feature_mapping[code])
    return fig
