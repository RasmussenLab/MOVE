# Load functions
from scipy.stats.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import seaborn as sns
from collections import defaultdict
from statsmodels.stats.multitest import multipletests
from functools import reduce

import matplotlib
import matplotlib.pyplot as plt
import random
import copy
import scipy
from scipy import stats
matplotlib.use('agg')
plt.style.use('seaborn-whitegrid')

import os, sys
import torch
import numpy as np
from torch.utils import data
import re 

from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
import itertools

from utils.data_utils import *
from analysis.analysis import *
from training.train import train_model_association
from utils.model_utils import *
from utils.visualization_utils import *

# Load MOVE specific funtions
path = "./"
sys.path.append(path + "src/")
import VAE_v2_1


data_dict = read_yaml('data')
cat_list, con_list, cat_names, con_names, headers_all, drug, drug_h = get_data(data_dict)


# Intilize variables
cuda = False

nLatents = [150, 200, 250, 300]

nepochs=1
lrate=0.0001
kldsteps=[15, 20, 25]
batchsteps=[50, 100, 150]
l = len(kldsteps)
rate = 20/l
# update = 1 + rate
repeats = 1
version = "v1"


def main():
    
    #train model
    train_model_association(path, cuda, nepochs, nLatents, con_list, cat_list, version, repeats, rate, kldsteps, batchsteps, lrate, drug)
    
    #load files 
    results = np.load(path + "results/results_" + version + ".npy", allow_pickle=True).item()
    recon_results = np.load(path + "results/results_recon_" + version + ".npy", allow_pickle=True).item()
    groups = np.load(path + "results/results_groups_" + version + ".npy", allow_pickle=True).item()
    
    mean_bas = np.load(path + "results/results_recon_mean_baseline_" + version + ".npy", allow_pickle=True).item()
    recon_results_1 = np.load(path + "results/results_recon_no_corr_" + version + ".npy", allow_pickle=True).item()
    cor_results = np.load(path + "wp2.2/sig_overlap/cor_results_" + version + ".npy", allow_pickle=True).item()

    recon_average = cal_reconstruction_change(recon_results)
    
    sig_hits, median_p_val = overlapping_hits(nLatents, cor_results, repeats, con_names, drug)
    
    all_hits, collected_overlap = identify_high_supported_hits(sig_hits, drug_h, version, path)
    
    report_values(path, sig_hits, median_p_val, drug_h, all_hits, con_names)
    
    recon_average_corr_new_all, recon_average_corr_all_indi_new = get_change_in_reconstruction(recon_average, groups, drug, drug_h, con_names, collected_overlap, sig_hits, np.concatenate(con_list, axis=-1), version, path)
    
    np.save(path + "results/results_confidence_recon_all_" + version + ".npy", recon_average_corr_new_all)
    np.save(path + "results/results_confidence_recon_all_indi_" + version + ".npy", recon_average_corr_all_indi_new)

    recon_average_corr_all_indi_new = np.load(path + "../../move/MOVE/results/results_confidence_recon_all_indi_" + version + ".npy", allow_pickle=True)

    up_down_list = ['baseline_target_metabolomics', 'baseline_untarget_metabolomics']         
    
    write_omics_results(path, up_down_list collected_overlap, recon_average_corr_new_all, headers_all)
    
    data_dict = read_yaml('data')
    con_dataset_names = data_dict['continuous_data_files']

    # con_dataset_names = ['Clinical_continuous', 'Diet_wearables','Proteomics','Targeted_metabolomics','Unargeted_metabolomics', 'Transcriptomics', 'Metagenomics']
    con_list_concat = np.concatenate(con_list, axis=1)
    make_files(collected_overlap, groups, con_list_concat, path, recon_average_corr_all_indi_new, con_names, con_dataset_names, drug_h, drug, all_hits, version = version)
    
    
    df_indi_var = get_inter_drug_variation(con_names, drug_h, recon_average_corr_all_indi_new, 
                                           groups, collected_overlap, drug, con_all, path)

    visualize_indi_var(df_indi_var, version)
    visualize_drug_similarity_across_all(recon_average_corr_new_all, drug_h, version)
    
    get_drug_similar_each_omics(con_names, con_dataset_names, all_hits, recon_average_corr_new_all, drug_h, version)

    
if __name__ == "__main__":
    main()