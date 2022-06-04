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
import VAE_v2_1

from move._utils.data_utils import *
from move._utils.visualization_utils import draw_boxplot
from move._training.train import optimize_stability
from move._analysis.analysis import get_top10_stability, calculate_latent



nHiddens = [1000]
nLatents = [50]
drop_outs = [0.1]
repeat = 5

nepochs = 2  #nepochs = 250
nl = 1
lrate=1e-3 #Added
kldsteps=[20, 30, 40, 90] #Added
batchsteps=[50, 100, 150, 200, 250, 300, 350, 400, 450] #Added
cuda = False

def main():
    embeddings, latents, con_recons, cat_recons, recon_acc = optimize_stability(nHiddens, nLatents, 
                                                                                drop_outs, repeat,
                                                                                nepochs, nl,
                                                                                lrate, kldsteps,
                                                                                batchsteps, cuda, 
                                                                                path, con_list, cat_list)
    
    stability_top10 = get_top10_stability(nHiddens, nLatents, drop_outs, repeat, nl, latents)
    
    stability_total, rand_index = calculate_latent(nHiddens, nLatents, drop_outs, repeat, nl, latents) # Todo add priting or smth
    
    
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


    
    

