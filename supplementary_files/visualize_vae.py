"""
This code plots the the trained autoencoder as a graph, where the edges and their color represent the weights.
If you don't know the input size, it is stated in logs/identify_associations as, e.g., 1343 in Model: VAE (1343 ⇄ 720 ⇄ 50).
The rest of the parameters can also be found in the config file for the identify associations task.

Args: 
-mp or --models_path: path to model weight files of the format model___.pt
-op or --output_path: path where the png figure will be saved
-in or --n_input: number of input nodes
-hi or --n_hidden: number of hidden nodes
-la or --n_latent: number of latent nodes
-re or --refit: refit number of the model we want to visualize

Returns:
    Png figure of the VAE's weights for a given refit of the model.

Example:

    python visualize_vae.py -mp /Users/______/Desktop/MOVE/tutorial/interim_data/models \\
                            -op /Users/______/Desktop/MOVE/tutorial/results/identify_associations/figures \\
                            -in 1343 \\
                            -hi 720 \\
                            -la 50 \\
                            -re 0

"""

import argparse
from pathlib import Path

from move.visualization.vae_visualization import plot_vae

parser = argparse.ArgumentParser(
    description="Plot weights and nodes for a trained autoencoder"
)
parser.add_argument(
    "-mp",
    "--models_path",
    metavar="mp",
    type=Path,
    required=True,
    help="path to model weight files of the format model___.pt",
)
parser.add_argument(
    "-op",
    "--output_path",
    metavar="op",
    type=Path,
    required=True,
    help="path where the png figure will be saved",
)
parser.add_argument(
    "-in",
    "--n_input",
    metavar="i",
    type=int,
    required=True,
    help="number of input nodes",
)
parser.add_argument(
    "-hi",
    "--n_hidden",
    metavar="h",
    type=int,
    required=True,
    help="number of hidden nodes",
)
parser.add_argument(
    "-la",
    "--n_latent",
    metavar="l",
    type=int,
    required=True,
    help="number of latent nodes",
)
parser.add_argument(
    "-re",
    "--refit",
    metavar="r",
    type=str,
    required=True,
    help="refit number of the model we want to visualize",
)

args = parser.parse_args()


plot_vae_base = plot_vae(
    args.models_path,
    args.output_path,
    f"model_{args.n_latent}_{args.refit}.pt",
    f"Vae's weights for refit {args.refit}",
    num_input=args.n_input,
    num_hidden=args.n_hidden,
    num_latent=args.n_latent,
    plot_edges=True,
)
