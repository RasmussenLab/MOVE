__all__ = ["plot_value_distributions"]

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from move.core.typing import FloatArray
from move.visualization.style import (
    DEFAULT_DIVERGING_PALETTE,
    DEFAULT_PLOT_STYLE,
    style_settings,
)


def plot_value_distributions(
    feature_values: FloatArray,
    style: str = "fast",
    nbins: int = 100,
    colormap: str = DEFAULT_DIVERGING_PALETTE,
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
        fig = plt.figure(layout="constrained", figsize=(7, 7))
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

        ax.plot_surface(x_val, y_val, np.array(histogram), cmap=colormap)
        ax.set_xlabel("Feature value")
        ax.set_ylabel("Feature ID number")
        ax.set_zlabel("Frequency")
        # ax.legend()
    return fig


def plot_reconstruction_diff(
    diff_array: FloatArray,
    vmin=None,
    vmax=None,
    style: str = DEFAULT_PLOT_STYLE,
    colormap: str = DEFAULT_DIVERGING_PALETTE,
) -> matplotlib.figure.Figure:
    """
    Plot the reconstruction differences as a heatmap.
    """
    with style_settings(style):
        if vmin is None:
            vmin = np.min(diff_array)
        elif vmax is None:
            vmax = np.max(diff_array)
        fig = plt.figure(layout="constrained", figsize=(7, 7))
        plt.imshow(diff_array, cmap=colormap, vmin=vmin, vmax=vmax)
        plt.xlabel("Feature")
        plt.ylabel("Sample")
        plt.colorbar()

    return fig


def plot_feature_association_graph(
    association_df, output_path, layout="circular", style: str = DEFAULT_PLOT_STYLE
):
    """
    This function plots a graph where each node corresponds to a feature and the edges
    represent the associations between features. Edge width represents the probability of
    said association, not the association's effect size.

    Input:
        association_df: pandas dataframe containing the following columns:
                            - feature_a: source node
                            - feature_b: target node
                            - p_value/bayes_score: edge weight
        output_path: Path object where the picture will be stored.

    Output:
        Feature_association_graph.png: picture of the graph

    """

    if "p_value" in association_df.columns:
        association_df["weight"] = 1 - association_df["p_value"]

    elif "proba" in association_df.columns:
        association_df["weight"] = association_df["proba"]

    elif "ks_distance" in association_df.columns:
        association_df["weight"] = association_df["ks_distance"]

    with style_settings(style):
        fig = plt.figure(figsize=(45, 45))
        G = nx.from_pandas_edgelist(
            association_df,
            source="feature_a_name",
            target="feature_b_name",
            edge_attr="weight",
        )

        nodes = list(G.nodes)

        datasets = association_df["feature_b_dataset"].unique()
        color_map = {
            dataset: (np.random.uniform(), np.random.uniform(), np.random.uniform())
            for dataset in datasets
        }
        node_dataset_map = {
            target_feature: dataset
            for (target_feature, dataset) in zip(
                association_df["feature_b_name"], association_df["feature_b_dataset"]
            )
        }

        if layout == "spring":
            pos = nx.spring_layout(G)
            with_labels = True
        elif layout == "circular":
            pos = nx.circular_layout(G)
            _ = [
                plt.text(
                    pos[node][0],
                    pos[node][1],
                    nodes[i],
                    rotation=(i / float(len(nodes))) * 360,
                    fontsize=10,
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                for i, node in enumerate(nodes)
            ]
            with_labels = False

        else:
            raise ValueError(
                "Graph layout (layout argument) must be either 'circular' or 'spring'."
            )

        nx.draw(
            G,
            pos=pos,
            with_labels=with_labels,
            node_size=2000,
            node_color=[
                (
                    color_map[node_dataset_map[feature]]
                    if feature in node_dataset_map.keys()
                    else "white"
                )
                for feature in G.nodes
            ],
            edge_color=list(nx.get_edge_attributes(G, "weight").values()),
            font_color="black",
            font_size=10,
            edge_cmap=matplotlib.colormaps["Purples"],
            connectionstyle="arc3, rad=1",
        )

        plt.tight_layout()
        fig.savefig(
            output_path / f"Feature_association_graph_{layout}.png", format="png"
        )


def plot_feature_mean_median(
    array: FloatArray, axis=0, style: str = DEFAULT_PLOT_STYLE
) -> matplotlib.figure.Figure:
    """
    Plot feature values together with the mean, median, min and max values at each array position.
    """
    with style_settings(style):
        fig = plt.figure(figsize=(15, 3))
        y = np.mean(array, axis=axis)
        y_2 = np.median(array, axis=axis)
        y_3 = np.max(array, axis=axis)
        y_4 = np.min(array, axis=axis)
        plt.plot(np.arange(len(y)), y, "bo", label="mean")
        plt.plot(np.arange(len(y_2)), y_2, "ro", label="median")
        plt.plot(np.arange(len(y_3)), y_3, "go", label="max")
        plt.plot(np.arange(len(y_4)), y_4, "yo", label="min")
        plt.legend()
        plt.xlabel("feature")
        plt.ylabel("mean/median/min/max")

    return fig


def plot_reconstruction_movement(
    baseline_recon: FloatArray,
    perturb_recon: FloatArray,
    k: int,
    style: str = DEFAULT_PLOT_STYLE,
) -> matplotlib.figure.Figure:
    """
    Plot, for each sample, the change in value from the unperturbed reconstruction to the perturbed reconstruction.
    Blue lines are left/negative shifts, red lines are right/positive shifts.

    Args:
        baseline_recon: baseline reconstruction array with s samples and k features (s,k).
        perturb_recon:  perturbed "                                                      "
        k: feature index. The shift (movement) of this feature's reconstruction will be plotted for all
           samples s.
    """
    with style_settings(style):
        # Feature changes
        fig = plt.figure(figsize=(25, 25))
        for s in range(np.shape(baseline_recon)[0]):
            plt.arrow(
                baseline_recon[s, k],
                s / 100,
                perturb_recon[s, k],
                0,
                length_includes_head=True,
                color=["r" if baseline_recon[s, k] < perturb_recon[s, k] else "b"][0],
            )
        plt.ylabel("Sample (e2)", size=40)
        plt.xlabel("Feature_value", size=40)
    return fig


def plot_cumulative_distributions(
    edges,
    hist_base,
    hist_pert,
    title,
    style: str = DEFAULT_PLOT_STYLE,
) -> matplotlib.figure.Figure:
    """
    Plot the cumulative distribution of the histograms for the baseline
    and perturbed reconstructions. This is useful to visually assess the
    magnitude of the shift corresponding to the KS score.

    """

    with style_settings(style):
        # Cumulative distribution:
        fig = plt.figure(figsize=(7, 7))
        plt.plot(
            (edges[:-1] + edges[1:]) / 2,
            np.cumsum(hist_base),
            color="blue",
            label="baseline",
            alpha=0.5,
        )
        plt.plot(
            (edges[:-1] + edges[1:]) / 2,
            np.cumsum(hist_pert),
            color="red",
            label="Perturbed",
            alpha=0.5,
        )

        plt.title(f"{title}.png")
        plt.xlabel("Feature value")
        plt.ylabel("Cumulative distribution")
        plt.legend()

    return fig


def plot_correlations(
    x, y, x_pol, y_pol, a2, a1, a, k, style: str = DEFAULT_PLOT_STYLE
):
    """
    Plot y vs x and the corresponding polynomial fit.
    """
    with style_settings(style):
        # Plot correlations
        fig = plt.figure(figsize=(7, 7))
        plt.plot(x, y, marker=".", lw=0, markersize=1, color="red")
        plt.plot(
            x_pol,
            y_pol,
            color="blue",
            label="{0:.2f}x^2 {1:.2f}x {2:.2f}".format(a2, a1, a),
            lw=1,
        )
        plt.plot(x_pol, x_pol, lw=1, color="k")
        plt.xlabel(f"Feature {k} baseline values ")
        plt.ylabel(f"Feature {k} baseline  value reconstruction")
        plt.legend()

    return fig


def get_2nd_order_polynomial(x_array, y_array, n_points=100):
    """
    Given a set of x an y values, find the 2nd oder polynomial fitting best the data.
    Returns:
        x_pol: x coordinates for the polynomial function evaluation.
        y_pol: y coordinates for the polynomial function evaluation.
    """
    a2, a1, a = np.polyfit(x_array, y_array, deg=2)

    x_pol = np.linspace(np.min(x_array), np.max(x_array), n_points)
    y_pol = a2 * x_pol**2 + a1 * x_pol + a

    return x_pol, y_pol, (a2, a1, a)
