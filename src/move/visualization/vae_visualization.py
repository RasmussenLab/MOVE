__all__ = ["visualize_vae"]

from pathlib import Path

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import torch

from move.core.typing import FloatArray
from move.visualization.style import DEFAULT_PLOT_STYLE, style_settings

def plot_vae(path: Path, filename: str, num_input: int, num_hidden: int, num_latent: int):

    """
    k input node index
    j hidden node index
    i latent node index
    """
    model_weights = torch.load(path / filename)
    G = nx.Graph()

    # Position of the layers:
    layer_distance = 2
    node_distance = 100
    latent_node_distance = 400



    ######### Adding nodes to the graph ###################
    # Bias nodes
    G.add_node("input_bias", pos = (-6*layer_distance,-3*node_distance))
    G.add_node("mu_bias", pos = (-3*layer_distance,(num_hidden+10)*node_distance), color = "red" )
    G.add_node("var_bias", pos = (-3*layer_distance,-10*node_distance), color = "green")
    G.add_node("out_bias", pos = (3*layer_distance,-3*node_distance))

    # Actual nodes
    for k in range(num_input):
        G.add_node(f"input_{k}", pos = (-6*layer_distance,k*node_distance-num_input*node_distance/2))
        G.add_node(f"output_{k}", pos = (6*layer_distance,k*node_distance-num_input*node_distance/2))
    for j in range(num_hidden):
        G.add_node(f"encoder_hidden_{j}", pos = (-3*layer_distance,j*node_distance-num_hidden*node_distance/2))
        G.add_node(f"decoder_hidden_{j}", pos = (3*layer_distance,j*node_distance-num_hidden*node_distance/2))
    for i in range(num_latent):
        G.add_node(f"mu_{i}", pos = (0*layer_distance,i*latent_node_distance))
        G.add_node(f"var_{i}", pos = (0*layer_distance,-i*latent_node_distance))
        G.add_node(f"sam_{i}", pos =(0.5*layer_distance,i*latent_node_distance-num_latent*latent_node_distance/2))

    ########## Adding weights to the graph #################
    for layer, values in model_weights.items():
        print(values.shape)
        if layer == "encoderlayers.0.weight":
            for k in range(values.shape[1]):     # input
                for j in range(values.shape[0]): # encoder_hidden
                    G.add_edge(f"input_{k}",f"encoder_hidden_{j}", weight=values.numpy()[j,k])
        
        elif layer == "encoderlayers.0.bias":
            for j in range(values.shape[0]): # encoder_hidden
                    G.add_edge(f"input_bias",f"encoder_hidden_{j}", weight=values.numpy()[j])

        elif layer == "mu.weight":
            for j in range(values.shape[1]): # encoder hidden
                for i in range(values.shape[0]): # mu
                    G.add_edge(f"encoder_hidden_{j}",f"mu_{i}", weight=values.numpy()[i,j])
            
        elif layer == "mu.bias":
            for i in range(values.shape[0]): # encoder_hidden
                    G.add_edge(f"mu_bias",f"mu_{i}", weight=values.numpy()[i])
        
        elif layer == "var.weight":
            for j in range(values.shape[1]): # encoder hidden
                for i in range(values.shape[0]): # var
                    G.add_edge(f"encoder_hidden_{j}",f"var_{i}", weight=values.numpy()[i,j])

        elif layer == "var.bias":
            for i in range(values.shape[0]): # encoder_hidden
                    G.add_edge(f"var_bias",f"var_{i}", weight=values.numpy()[i])

        # Sampled layer from mu and var:
        elif layer == "decoderlayers.0.weight":
            for i in range(values.shape[1]):     # sampled latent
                for j in range(values.shape[0]): # decoder_hidden
                    G.add_edge(f"sam_{i}",f"decoder_hidden_{j}", weight=values.numpy()[j,i])
        

        elif layer == "out.weight":
            for j in range(values.shape[1]):     # decoder_hidden 
                for k in range(values.shape[0]): # output
                    G.add_edge(f"output_{k}",f"decoder_hidden_{j}", weight=values.numpy()[k,j])

        elif layer == "out.bias":
            for k in range(values.shape[0]): # output
                G.add_edge(f"out_bias",f"output_{k}", weight=values.numpy()[k])


    fig = plt.figure(figsize=(30,90))
    pos = nx.get_node_attributes(G,"pos")
    nx.draw(G, 
            pos=pos,
            with_labels=False,
            node_size= 10,
            #node_color=["gold" if "meta" in feature else "purple" for feature in G.nodes],
            edge_color=list(nx.get_edge_attributes(G,"weight").values()),
            font_color= "black",
            font_size=10,
            edge_cmap=matplotlib.colormaps["bwr"])

    #plt.tight_layout()
    fig.savefig(path / f"VAE.png", format = "png", dpi = 300)


#######

path = Path("/Users/wjq311/Desktop/MOVE/tutorial/results_cont/latent_space")
filename = "model.pt"

plot_vae(path, filename, num_input=280, num_hidden=560, num_latent=35)

