__all__ = ["plot_value_distributions"]

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import networkx as nx
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

def plot_reconstruction_diff(diff_array: FloatArray, vmin=None, vmax=None) -> matplotlib.figure.Figure:
    """
    Plot the reconstruction differences as a heatmap
    """
    #colstep = 10
    #samplestep = 10

    if vmin == None:
        vmin = np.min(diff_array)
    elif vmax == None:
         vmax = np.max(diff_array)
    fig = plt.figure(layout="constrained", figsize=(10,10))
    plt.imshow(diff_array, cmap ="bwr", vmin=vmin, vmax=vmax)
    plt.xlabel("Feature")
    plt.ylabel("Sample")
    plt.colorbar()
    #plt.xticks(ticks = np.arange(0,np.shape(diff_array)[1],colstep), labels = colnames[::colstep], rotation = 90)
    #plt.yticks(ticks = np.arange(0,np.shape(diff_array)[0],samplestep), labels = samplenames[::colstep])
    return fig



def plot_feature_association_graph(association_df, output_path, layout="circular"):
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
        association_df["weight"] = 1- association_df["p_value"]

    else:
        association_df["weight"] = association_df["proba"]
        
    fig = plt.figure(figsize=(45,45))
    G = nx.from_pandas_edgelist(association_df,
                                source="feature_a_name",
                                target="feature_b_name",
                                edge_attr="weight")

    nodes = list(G.nodes)

    if layout == "spring":
        pos = nx.spring_layout(G)
        with_labels = True
    elif layout == "circular":
        pos = nx.circular_layout(G)
        texts = [plt.text(pos[node][0],pos[node][1],nodes[i],rotation=(i/float(len(nodes)))*360,fontsize=10,horizontalalignment='center',verticalalignment='center') for i,node in enumerate(nodes)]
        with_labels = False
    #pos = nx.spring_layout(G, weight="weight")

    nx.draw(G, 
            pos=pos,
            with_labels=with_labels,
            node_size= 2000,
            node_color=["gold" if "meta" in feature else "purple" for feature in G.nodes],
            edge_color=list(nx.get_edge_attributes(G,"weight").values()),
            font_color= "black",
            font_size=10,
            edge_cmap=matplotlib.colormaps["Purples"],
            connectionstyle = "arc3, rad=1")

 
    plt.tight_layout()
    fig.savefig(output_path / f"Feature_association_graph_{layout}.png", format = "png")

def plot_feature_mean_median(array: FloatArray, axis=0) ->matplotlib.figure.Figure:

    fig = plt.figure(figsize=(15,3))
    y = np.mean(array, axis=axis)
    y_2 = np.median(array, axis=axis)
    y_3 = np.max(array, axis=axis)
    y_4 = np.min(array, axis=axis)
    plt.plot(np.arange(len(y)), y, "bo", label= "mean" )
    plt.plot(np.arange(len(y_2)), y_2, "ro", label="median")
    plt.plot(np.arange(len(y_3)), y_3, "go", label="max")
    plt.plot(np.arange(len(y_4)), y_4, "yo", label="min")
    plt.legend()
    plt.xlabel("feature")
    plt.ylabel("mean/median/min/max")
    
    return fig

def get_2nd_order_polynomial(x_array,y_array, n_points=100):
    """ 
    Given a set of x an y values, find the 2nd oder polynomial fitting best the data.
    Returns:
        x_pol: x coordinates for the polynomial function evaluation.
        y_pol: y coordinates for the polynomial function evaluation.
    """
    a2, a1, a = np.polyfit(x_array,y_array,deg=2)

    x_pol = np.linspace(np.min(x_array),np.max(x_array),n_points)
    y_pol = np.array([a2*x*x+a1*x+a for x in x_pol])

    return x_pol,y_pol, (a2,a1,a)
