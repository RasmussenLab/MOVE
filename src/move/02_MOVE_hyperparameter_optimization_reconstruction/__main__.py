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

import VAE_v2_1
import os
import yaml 

from move._utils.data_utils import *
from move._utils.model_utils import *
from move._utils.visualization_utils import *
from move._training.train import optimize_reconstruction

import hydra 
from move.conf.schema import MOVEConfig
from move._utils.data_utils import get_data

@hydra.main(config_path="../conf", config_name="main")
def main(config: MOVEConfig): 
    
    #Get needed variables
    cuda = config.training.cuda
    path = config.data.processed_data_path
    nHiddens = config.tuning_reconstruction.nHiddens
    nLatents = config.tuning_reconstruction.nLatents
    nLayers = config.tuning_reconstruction.nLayers
    nDropout = config.tuning_reconstruction.nDropout
    nBeta = config.tuning_reconstruction.nBeta
    batch_sizes = config.tuning_reconstruction.batch_sizes
    nepochs = config.training.num_epochs
    repeat = 1
    lrate = config.training.lr
    kld_steps = config.training.kld_steps
    batch_steps = config.training.batch_steps
    patience = config.training.patience
    path = config.data.processed_data_path
    categorical_names = config.model.categorical_names
    continuous_names = config.model.continuous_names
    data_of_interest = config.data.data_of_interest
    
    #Get the data
    cat_list, con_list, cat_names, con_names, headers_all, drug, drug_h = get_data(path, categorical_names, continuous_names, data_of_interest)
    
    #Perform hyperparameter tuning
    optimize_reconstruction(nHiddens, nLatents, 
                            nLayers, nDropout, 
                            nBeta, batch_sizes, 
                            nepochs, repeat, 
                            lrate, kld_steps, 
                            batch_steps, patience, 
                            cuda, path, 
                            cat_list, con_list)

    #Visualize the data
    visualize_likelihood(nLayers, nHiddens, nDropout, nBeta, nLatents, likelihood_tests)
    visualize_recon_acc(nLayers, nHiddens, nDropout, nBeta, nLatents, recon_acc_tests, 'test')
    visualize_recon_acc(nLayers, nHiddens, nDropout, nBeta, nLatents, recon_acc, 'train')    


if __name__ == "__main__":
    main()
