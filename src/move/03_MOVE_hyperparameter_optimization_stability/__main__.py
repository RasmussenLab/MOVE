# Import functions
import os, sys
import torch
import numpy as np
from torch.utils import data

from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

# import umap
import umap.umap_ as umap

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from scipy.stats.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import seaborn as sns
import matplotlib
import re
import random
from collections import defaultdict
import itertools 
# import tqdm   

path="./"
sys.path.append(path + "src/")
import move.VAE_v2_1

from move._utils.data_utils import *
from move._utils.visualization_utils import draw_boxplot
from move._training.train import optimize_stability
from move._analysis.analysis import get_top10_stability, calculate_latent

import hydra 
from move.conf.schema import MOVEConfig
from move._utils.data_utils import get_data

def get_list_value(*args):
    arg_tuple = [arg[0] if len(arg) == 1 else arg for arg in args]
    return(arg_tuple)

@hydra.main(config_path="../conf", config_name="main")
def main(config: MOVEConfig): 
    #Get needed variables
    cuda = config.training.cuda
    path = config.data.processed_data_path
    nHiddens = config.tuning_stability.nHiddens
    nLatents = config.tuning_stability.nLatents
    nLayers = config.tuning_stability.nLayers
#     nLayers = 1
    nDropout = config.tuning_stability.nDropout
    nBeta = config.tuning_stability.nBeta
    batch_sizes = config.tuning_stability.batch_sizes
    nepochs = config.training.num_epochs
    repeat = config.training.repeat
    lrate = config.training.lr
    kld_steps = config.training.kld_steps
    batch_steps = config.training.batch_steps
    patience = config.training.patience
    path = config.data.processed_data_path
    categorical_names = config.model.categorical_names
    continuous_names = config.model.continuous_names
    data_of_interest = config.data.data_of_interest
    
    nLayers, batch_sizes = get_list_value(nLayers, batch_sizes)
    
    #Get the data
    cat_list, con_list, cat_names, con_names, headers_all, drug, drug_h = get_data(path, categorical_names, continuous_names, data_of_interest)

    
    embeddings, latents, con_recons, cat_recons, recon_acc = optimize_stability(nHiddens, nLatents, 
                                                                                nDropout, repeat,
                                                                                nepochs, nLayers,
                                                                                batch_sizes, lrate, 
                                                                                kld_steps, batch_steps, 
                                                                                cuda, path, 
                                                                                con_list, cat_list)
    
    stability_top10 = get_top10_stability(nHiddens, nLatents, drop_outs, repeat, nLayers, latents)
    
    stability_total, rand_index = calculate_latent(nHiddens, nLatents, nDropout, repeat, nLayers, latents) # Todo add priting or smth
    
    
    draw_boxplot(df=stability_top10,
             title_text='Difference across replicationes in cosine similarity of ten closest neighbours in first iteration',
             y_label_text="Average change",
             save_fig_name="stability_top10")
    
    
    draw_boxplot(df=stability_total,
             title_text='Difference across replicationes in cosine similarity compared to first iteration',
             y_label_text="Average change",
             save_fig_name="stability_all")
    
    draw_boxplot(df=rand_index,
             title_text='Rand index across replicationes compared to first iteration',
             y_label_text="Rand index",
             save_fig_name="rand_index_all")

if __name__ == "__main__":
    main()


    
    

