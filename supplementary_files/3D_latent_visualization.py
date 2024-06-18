"""
This code is meant to 1) compress the latent representation of the data contained in latent_location.npy 
with shape (n_samples,n_features,n_perturbations) to 3 UMAP dimensions.
Then, the location of the samples in 3D-UMAP is plotted, color coding by a feature of interest. The movement
of the samples when perturbing said figure is shown using the same UMAP projection as the baseline.

Args:
    -lp or --latent_path: path to latent numpy array of shape (n_samples,n_features,n_perturbations)
    -dp or --data_path: path to original datasets, in the example is interim_path
    -ds or --dataset: name of the perturbed dataset
    -foi or --feature_of_interest: feature that we want to perturb and visualize

Returns:
    figure folder inside latent_path/ with figures and gifs depicting the latent space distribution of
    the feature of interest (perturbed_feature.gif) and the movement that samples undergo when perturbing 
    said feature (arrows.gif)

Example: 
    args.latent_path = Path("/Users/_____/Desktop/MOVE/tutorial/results/identify_associations")
    args.data_path = Path("/Users/_____/Desktop/MOVE/tutorial/interim_data")
    args.dataset = "ibd.mbx"
    args.feature_of_interest = "C20 carnitine"

How to run:
    Example:
    1) go to the folder where this file is located:
    cd /Users/____/Desktop/MOVE/supplementary_files
    2) type the following substituting the fields for your files
    python 3D_latent_visualization.py -lp /Users/____/Desktop/MOVE/tutorial/results/identify_associations \\
                                      -dp /Users/____/Desktop/MOVE/tutorial/interim_data \\
                                      -ds ibd.mbx \\
                                      -foi="C20 carnitine"  
                                      
Note: UMAP must be installed, which can be done by running:
    pip install umap-learn
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from PIL import Image

from move.visualization.latent_space import plot_3D_latent_and_displacement

parser = argparse.ArgumentParser(
    description="Read latent space matrix file to plot it in 3D"
)
parser.add_argument(
    "-lp",
    "--latent_path",
    metavar="lp",
    type=Path,
    required=True,
    help="path to latent numpy array (n_samples,n_features,n_perturbations)",
)
parser.add_argument(
    "-dp",
    "--data_path",
    metavar="dp",
    type=Path,
    required=True,
    help="path to original datasets, interim_path",
)
parser.add_argument(
    "-ds",
    "--dataset",
    metavar="ds",
    type=str,
    required=True,
    help="name of the perturbed dataset",
)
parser.add_argument(
    "-foi",
    "--feature_of_interest",
    metavar="foi",
    type=str,
    required=True,
    help="feature that we want to perturb",
)
args = parser.parse_args()

figure_path = Path(args.latent_path / "figures")
figure_path.mkdir(exist_ok=True, parents=True)

perturbed_dataset = np.load(args.data_path / f"{args.dataset}.npy")
perturbed_features = list(np.load(args.latent_path / "perturbed_features_list.npy"))

latent_matrix = np.load(args.latent_path / "latent_location.npy")
trans = umap.UMAP(random_state=42, n_components=3).fit(latent_matrix[:, :, -1])
embedding = trans.embedding_

if args.feature_of_interest not in perturbed_features:
    raise ValueError(" Feature of interest not in perturbed dataset")

i = perturbed_features.index(args.feature_of_interest)

new_embedding = trans.transform(latent_matrix[:, :, i])

# # Plot latent space:
pic_num = 0
n_pictures = 100
for azimuth, altitude in zip(
    np.linspace(-45, 45, n_pictures), np.linspace(15, 45, n_pictures)
):
    fig = plot_3D_latent_and_displacement(
        embedding,
        new_embedding,
        feature_values=perturbed_dataset[:, i],
        feature_name=f"Sample movement",
        show_baseline=False,
        show_perturbed=False,
        show_arrows=True,
        step=1,
        altitude=altitude,
        azimuth=azimuth,
    )

    fig.savefig(figure_path / f"3D_latent_movement_{pic_num}_arrows.png", dpi=100)
    plt.close(fig)

    fig = plot_3D_latent_and_displacement(
        embedding,
        new_embedding,
        feature_values=perturbed_dataset[:, i],
        feature_name=f"Feature {args.feature_of_interest}",
        show_baseline=True,
        show_perturbed=False,
        show_arrows=False,
        altitude=altitude,
        azimuth=azimuth,
    )
    fig.savefig(
        figure_path / f"3D_latent_movement_{pic_num}_perturbed_feature.png", dpi=100
    )
    plt.close(fig)
    pic_num += 1

for plot_type in ["arrows", "perturbed_feature"]:
    frames = [
        Image.open(figure_path / f"3D_latent_movement_{pic_num}_{plot_type}.png")
        for pic_num in range(n_pictures)
    ]  # sorted(glob.glob("*3D_latent*"))]
    frames[0].save(
        figure_path / f"{plot_type}.gif",
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=75,
        loop=0,
    )
