__all__ = ["plot_vae"]

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.cm as cm
import matplotlib.figure
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch


def plot_vae(
    path: Path,
    savepath: Path,
    filename: str,
    title: str,
    num_input: int,
    num_hidden: int,
    num_latent: int,
    plot_edges=True,
    input_sample: Optional[torch.Tensor] = None,
    output_sample: Optional[torch.Tensor] = None,
    mu: Optional[torch.Tensor] = None,
    logvar: Optional[torch.Tensor] = None,
) -> matplotlib.figure.Figure:
    """
    This function is aimed to visualize MOVE's architecture.

    Args:
        path: path where the trained model with defined weights is to be found
        filename: name of the model
        title: title of the figure
        num_input: number of input nodes
        num_hidden: number of output nodes
        num_latent: number of latent nodes
        plot_edges: plot edges, i.e. connections with assigned weights between nodes
        input_sample: array with input values to fill with a mapped color value
        output_sample : "      " output " "
        mu: "                  " mean (latent) "     "
        logvar: "              " log variance  "     "

    Returns:
        figure

    Notes:
        k: input node index
        j: hidden node index
        i: latent node index
    """
    model_weights = torch.load(path / filename)
    G = nx.Graph()

    # Position of the layers:
    layer_distance = 10
    node_distance = 550
    latent_node_distance = 550
    latent_sep = 5 * latent_node_distance

    # Adding nodes to the graph ##############################
    # Bias nodes
    G.add_node(
        "input_bias",
        pos=(-6 * layer_distance, -3 * node_distance - num_input * node_distance / 2),
        color=0.0,
    )
    G.add_node(
        "mu_bias",
        pos=(
            -3 * layer_distance,
            (num_hidden + 3) * node_distance
            - num_hidden * node_distance / 2
            + latent_sep / 2,
        ),
        color=0.0,
    )
    G.add_node(
        "var_bias",
        pos=(
            -3 * layer_distance,
            -3 * node_distance - num_hidden * node_distance / 2 - latent_sep / 2,
        ),
        color=0.0,
    )
    G.add_node(
        "sam_bias",
        pos=(
            0.5 * layer_distance,
            -3 * latent_node_distance - num_latent * latent_node_distance / 2,
        ),
        color=0.0,
    )
    G.add_node(
        "out_bias",
        pos=(3 * layer_distance, -3 * node_distance - num_hidden * node_distance / 2),
        color=0.0,
    )

    # Actual nodes
    for k in range(num_input):
        G.add_node(
            f"input_{k}",
            pos=(
                -6 * layer_distance,
                k * node_distance - num_input * node_distance / 2,
            ),
            color=[input_sample[k] if input_sample is not None else 0.0][0],
        )
        G.add_node(
            f"output_{k}",
            pos=(6 * layer_distance, k * node_distance - num_input * node_distance / 2),
            color=[output_sample[k] if output_sample is not None else 0.0][0],
        )
    for j in range(num_hidden):
        G.add_node(
            f"encoder_hidden_{j}",
            pos=(
                -3 * layer_distance,
                j * node_distance - num_hidden * node_distance / 2,
            ),
            color=0.0,
        )
        G.add_node(
            f"decoder_hidden_{j}",
            pos=(
                3 * layer_distance,
                j * node_distance - num_hidden * node_distance / 2,
            ),
            color=0.0,
        )
    for i in range(num_latent):
        G.add_node(
            f"mu_{i}",
            pos=(0 * layer_distance, i * latent_node_distance + latent_sep / 2),
            color=[mu[i] if mu is not None else 0.0][0],
        )
        G.add_node(
            f"var_{i}",
            pos=(0 * layer_distance, -i * latent_node_distance - latent_sep / 2),
            color=[np.exp(logvar[i] / 2) if logvar is not None else 0.0][0],
        )
        G.add_node(
            f"sam_{i}",
            pos=(
                0.5 * layer_distance,
                i * latent_node_distance - num_latent * latent_node_distance / 2,
            ),
            color=0.0,
        )

    # Adding weights to the graph #########################

    if plot_edges:
        for layer, values in model_weights.items():
            if layer == "encoderlayers.0.weight":
                for k in range(values.shape[1]):  # input
                    for j in range(values.shape[0]):  # encoder_hidden
                        G.add_edge(
                            f"input_{k}",
                            f"encoder_hidden_{j}",
                            weight=values.numpy()[j, k],
                        )

            elif layer == "encoderlayers.0.bias":
                for j in range(values.shape[0]):  # encoder_hidden
                    G.add_edge(
                        "input_bias", f"encoder_hidden_{j}", weight=values.numpy()[j]
                    )

            elif layer == "mu.weight":
                for j in range(values.shape[1]):  # encoder hidden
                    for i in range(values.shape[0]):  # mu
                        G.add_edge(
                            f"encoder_hidden_{j}",
                            f"mu_{i}",
                            weight=values.numpy()[i, j],
                        )

            elif layer == "mu.bias":
                for i in range(values.shape[0]):  # encoder_hidden
                    G.add_edge("mu_bias", f"mu_{i}", weight=values.numpy()[i])

            elif layer == "var.weight":
                for j in range(values.shape[1]):  # encoder hidden
                    for i in range(values.shape[0]):  # var
                        G.add_edge(
                            f"encoder_hidden_{j}",
                            f"var_{i}",
                            weight=values.numpy()[i, j],
                        )

            elif layer == "var.bias":
                for i in range(values.shape[0]):  # encoder_hidden
                    G.add_edge("var_bias", f"var_{i}", weight=values.numpy()[i])

            # Sampled layer from mu and var:
            elif layer == "decoderlayers.0.weight":
                for i in range(values.shape[1]):  # sampled latent
                    for j in range(values.shape[0]):  # decoder_hidden
                        G.add_edge(
                            f"sam_{i}",
                            f"decoder_hidden_{j}",
                            weight=values.numpy()[j, i],
                        )

            # Sampled layer from mu and var:
            elif layer == "decoderlayers.0.bias":
                for j in range(values.shape[0]):  # decoder_hidden
                    G.add_edge(
                        "sam_bias", f"decoder_hidden_{j}", weight=values.numpy()[j]
                    )

            elif layer == "out.weight":
                for j in range(values.shape[1]):  # decoder_hidden
                    for k in range(values.shape[0]):  # output
                        G.add_edge(
                            f"output_{k}",
                            f"decoder_hidden_{j}",
                            weight=values.numpy()[k, j],
                        )

            elif layer == "out.bias":
                for k in range(values.shape[0]):  # output
                    G.add_edge("out_bias", f"output_{k}", weight=values.numpy()[k])

    fig = plt.figure(figsize=(60, 60))
    pos = nx.get_node_attributes(G, "pos")
    color = list(nx.get_node_attributes(G, "color").values())
    edge_color = list(nx.get_edge_attributes(G, "weight").values())
    edge_width = list(nx.get_edge_attributes(G, "weight").values())

    edge_cmap = matplotlib.colormaps["seismic"]
    node_cmap = matplotlib.colormaps["seismic"]

    abs_max = np.max([abs(np.min(color)), abs(np.max(color))])
    abs_max_edge = np.max([abs(np.min(edge_color)), abs(np.max(edge_color))])

    _ = cm.ScalarMappable(
        cmap=node_cmap, norm=matplotlib.colors.Normalize(vmin=-abs_max, vmax=abs_max)
    )
    sm_edge = cm.ScalarMappable(
        cmap=edge_cmap,
        norm=matplotlib.colors.Normalize(vmin=-abs_max_edge, vmax=abs_max_edge),
    )

    nx.draw(
        G,
        pos=pos,
        with_labels=True,
        node_size=100,
        node_color=color,
        edge_color=edge_color,
        width=edge_width,
        font_color="black",
        font_size=10,
        edge_cmap=edge_cmap,
        cmap=node_cmap,
        vmin=-abs_max,
        vmax=abs_max,
    )

    # plt.colorbar(sm_node, label="Node value", shrink = .2)
    plt.colorbar(sm_edge, label="Edge value", shrink=0.2)
    plt.tight_layout()
    fig.savefig(savepath / f"{title}.png", format="png", dpi=200)
    return fig
