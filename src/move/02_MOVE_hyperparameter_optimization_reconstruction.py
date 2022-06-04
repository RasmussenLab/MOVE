# Import functions
import os, sys
import torch
import numpy as np
from torch.utils import data
import copy

from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

import umap
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from scipy.stats.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import seaborn as sns
import matplotlib
import re
import random
from collections import defaultdict
import itertools 
import tqdm   

path="./"
sys.path.append(path + "src/")
import VAE_v2_1
import os
import yaml 

from utils.data_utils import *
from utils.model_utils import *
from utils.visualization_utils import *
from training.train import optimize_reconstruction


data_dict = read_yaml('data')
params_dict = read_yaml('02_tune_reconstruction')

cuda = params_dict['cuda']
path = params_dict['path']
nHiddens = params_dict['nHiddens']
nLatents = params_dict['nLatents']
nLayers = params_dict['nLayers']
nDropout = params_dict['nDropout']
nBeta = params_dict['nBeta']
batch_sizes = params_dict['batch_sizes']
nepochs = params_dict['nepochs']
lrate = float(params_dict['lrate'])
# repeat = params_dict['repeat']
patience = params_dict['patience']

repeat = 1
cat_list, con_list, cat_names, con_names, headers_all, drug, drug_h = get_data(data_dict)

kldsteps = [20, 30, 40, 90]
batchsteps = [50, 100, 150, 200, 250, 300, 350, 400, 450] 

def main():
    
    optimize_reconstruction(nHiddens, nLatents, 
                            nLayers, nDropout, 
                            nBeta, batch_sizes, 
                            nepochs, repeat, 
                            lrate, kldsteps, 
                            batchsteps, patience, 
                            cuda, path, 
                            cat_list, con_list)


    visualize_likelihood(nLayers, nHiddens, nDropout, nBeta, nLatents, likelihood_tests)
    visualize_recon_acc(nLayers, nHiddens, nDropout, nBeta, nLatents, recon_acc_tests, 'test')
    visualize_recon_acc(nLayers, nHiddens, nDropout, nBeta, nLatents, recon_acc, 'train')    


# Kinda ranking: reconstruction loss - 
# Then stability

# Rank by loss and then by stability

# Epochs - mean of all and round it up
# 02: Select the top combinations.

if __name__ == "__main__":
    main()
