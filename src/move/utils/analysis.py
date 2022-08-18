import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import umap.umap_ as umap

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

from scipy.stats.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf
import itertools 
from tqdm import tqdm
from collections import Counter, defaultdict
from scipy import stats

from move.utils.data_utils import initiate_default_dicts
from move.utils import dataloaders
from move.models import vae


def get_top10_stability(nHiddens, nLatents, drop_outs, nLayers, repeat, latents, batch_sizes, nBeta):
    '''
    Calculates stability focusing on the top 10 closest neigbour for each individual.
    
    inputs:
        nHiddens: a list with integers with the number of neurons in hidden layers
        nLatents: a list with integers with a size of the latent dimension
        drop_outs: a list with floats with dropout probabilities applied after each nonlinearity in encoder and decoder
        nLayers: a list with integers with the number of layers
        repeat: integer of the number of times to train the model with the same configuration
        latents: Defaultdict. Keys: set of hyperparameter values; values: np.array of VAE latent representation of input dataset
        batch_sizes: a list with ints with batch sizes
        nBeta: a list with floats with beta values (Multiplies KLD by the inverse of this value)
    returns:
        stability_top10: Defaultdict. Keys: set of hyperparameter values; values:TODO: list of floats of mean positional differences
        stability_top10_df: list of dicts, with hyperparameter values and mean positional difference
    '''

    npatient = list(latents.values())[0][0].shape[0]
    top10_changes, stability_top10 = initiate_default_dicts(0, 2) 
    
    stability_top10_df = [] 
        
    iters = itertools.product(nHiddens, nLatents, nLayers, drop_outs, nBeta)
    for nHidden, nLatent, nl, drop, b in iters:
        
        max_pos_values_init = list()
        old_sum_max = list()
        
        name = str([nHidden] * nl) + "+" + str(nLatent) + ", do: " + str(drop) +", b: " + str(b)

        print(name)
        top10_changes[name] = [ [] for i in range(npatient) ]
     
        for r in range(repeat):
            cos_sim = cosine_similarity(latents[name][r])
            corr = pd.DataFrame(cos_sim)
                
            step = list()
            for index, row in corr.iterrows():
                if r == 0:
                    max_pos = np.asarray(row.argsort()[11:][::-1][1:11])
                    max_pos_values_init.append(max_pos)
                    #summed_max = np.mean(row[max_pos])
                    old_sum_max.append(row[max_pos])
                else:
                    old_pos = max_pos_values_init[index]
                    old_sum = old_sum_max[index]
                    #summed_max = np.mean(row[old_pos])
                    top10_changes[name][index].append(np.mean(abs(old_sum - row[old_pos])))
                    step.append(np.mean(abs(old_sum - row[old_pos])))
            print(r)
            if r != 0:
                stability_top10[name].append(np.mean(step))
                stability_top10_df.append({
                    'num_hidden': nHidden,
                    'num_latent': nLatent,
                    'num_layers': nl,
                    'dropout': drop,
                    'beta': b,
                    'batch_sizes': batch_sizes, 
                    'repeats': r,
                    'difference': np.mean(step)
                    })
    stability_top10_df = pd.DataFrame(stability_top10_df)
    return(stability_top10, stability_top10_df)


def calculate_latent(nHiddens, nLatents, drop_outs, repeat, nLayers, nBeta, latents, batch_sizes):
    '''
    TODO: Calculates stability and 
    
    Inputs:
        nHiddens: a list with integers with the number of neurons in hidden layers
        nLatents: a list with integers with a size of the latent dimension
        drop_outs: a list with floats with dropout probabilities applied after each nonlinearity in encoder and decoder
        repeat: integer of the number of times to train the model with the same configuration
        nLayers: a list with integers with the number of layers
        nBeta: a list with floats with beta values (Multiplies KLD by the inverse of this value)
        latents: Defaultdict. Keys: set of hyperparameter values; values: np.array of VAE latent representation of input dataset
        batch_sizes: a list with ints with batch sizes
    returns:
        stability_total:  Defaultdict. Keys: set of hyperparameter values; values:TODO: list of floats of mean positional differences
        rand_index: Defaultdict. Keys: set of hyperparameter values; values:TODO: list of floats of mean positional differences
        stability_total_df: TODO: pd.DataFrame, with hyperparameter values and rand_inde mean positional difference
    '''
    

    npatient = list(latents.values())[0][0].shape[0]
    total_changes, stability_total, rand_index = initiate_default_dicts(0, 3)
    
    stability_total_df = [] 
    
    iters = itertools.product(nHiddens, nLatents, nLayers, drop_outs, nBeta)
    for nHidden, nLatent, nl, drop, b in iters:
       
        pos_values_init = list() 
        old_rows = list()
          
        name = str([nHidden] * nl) + "+" + str(nLatent) + ", do: " + str(drop) +", b: " + str(b)
        total_changes[name] = [ [] for i in range(npatient) ]

        for r in range(repeat):
            cos_sim = cosine_similarity(latents[name][r])

            corr = pd.DataFrame(cos_sim)
            step = list()
            for index, row in corr.iterrows():
                if r == 0:
                    max_pos = np.asarray(row.argsort()[:][::-1][1:])
                    pos_values_init.append(max_pos)
                    old_rows.append(row[max_pos])
                else:
                    old_pos = pos_values_init[index]
                    old_row = old_rows[index]
                    total_changes[name][index].append(np.mean(abs(old_row - row[old_pos])))
                    step.append(np.mean(abs(old_row - row[old_pos])))
            
            if r == 0:
                kmeans = KMeans(n_clusters=4)
                kmeans = kmeans.fit(latents[name][r])
                true_labels = kmeans.predict(latents[name][r])                
            else:
                kmeans = KMeans(n_clusters=4)
                rand_tmp = []
                for i in range(0,100):
                    kmeans = kmeans.fit(latents[name][r])
                    labels = kmeans.predict(latents[name][r])
                    rand_tmp.append(adjusted_rand_score(true_labels, labels))

                rand_index[name].append(np.mean(rand_tmp))
                stability_total[name].append(np.mean(step))
                
                stability_total_df.append({
                    'num_hidden': nHidden,
                    'num_latent': nLatent,
                    'num_layers': nl,
                    'dropout': drop,
                    'beta': b,
                    'batch_sizes': batch_sizes, 
                    'repeats': r,
                    'difference': np.mean(step),
                    'rand_index': np.mean(rand_tmp)
                    })
                

    stability_total = pd.DataFrame(stability_total)
    stability_total_df = pd.DataFrame(stability_total_df)
    return(stability_total, rand_index, stability_total_df)


def get_latents(best_model, train_loader, kld_w=1):
    '''
    Returns latent representations of the model predictions
    
    Inputs:
        best_model: model object that had lowest loss on tes
        train_loader: Dataloader of training set
        kld_w: float of KLD weight
    Returns:
        latent: np.array of VAE latent representation of input dataset
        latent_var: np.array of VAE latent representation of input dataset (sigma values)
        cat_recon: np.array of VAE model's reconstructions for categorical data
        cat_class: np.array of ordinally encoded input categorical data (-1 if it is missing)
        con_recon:  np.array of VAE model's reconstructions for continuous data
        loss: float of loss of VAE model's predictions
        likelihood: float of reconstruction losses (BCE for categorical datapoints and SSE for continuous datapoints)
    '''
     
    # Extracting the latent space
    train_test_loader = DataLoader(dataset=train_loader.dataset, batch_size=1, 
                                             drop_last=False, shuffle=False, 
                                             pin_memory=train_loader.pin_memory) 

    latent, latent_var, cat_recon, cat_class, \
    con_recon, loss, likelihood = best_model.latent(train_test_loader, 1)

    con_recon = np.array(con_recon)
    con_recon = torch.from_numpy(con_recon)
     
    return latent, latent_var, cat_recon, cat_class, con_recon, loss, likelihood


def calc_categorical_reconstruction_acc(cat_shapes, cat_class, cat_recon):
    '''
    Calculates reconstruction accuracy for categorical data
    
    Inputs:
        cat_shapes: list of tuple (npatient, nfeatures, ncategories) corresponding to categorical data shapes.
        cat_class: np.array of ordinally encoded input categorical data (-1 if it is missing)
        cat_recon: np.array of VAE model's reconstructions for continuous data
    Returns:
        cat_total_recon: list of floats (from 0 to 1), which corresponds to the fraction of how many samples were correctly reconstructed
        
    '''
   
    # Calculate the categorical reconstruction accuracy
    cat_true_recon = []
    cat_total_recon = []
    pos = 0
    for s in cat_shapes:
        n = s[1]
        cat_class_tmp = cat_class[:,pos:(n + pos)]
        cat_recon_tmp = cat_recon[:,pos:(n + pos)]

        missing_cat = cat_recon_tmp[cat_class_tmp == -1]
        diff_cat = cat_class_tmp - cat_recon_tmp

        diff_cat[diff_cat != 0] = -1
        true_cat = diff_cat[diff_cat == 0]
        false_cat = diff_cat[diff_cat != 0]

        cat_true = len(true_cat)/(float(diff_cat.size) - missing_cat.size)
        cat_true_recon.append(cat_true)
        diff_cat[diff_cat == 0] = 1
        diff_cat[diff_cat != 1] = 0
        cat_total_recon.append(np.count_nonzero(diff_cat, 1) / diff_cat.shape[1])
        pos += n
    return(cat_total_recon)


def calc_continuous_reconstruction_acc(con_shapes, con_recon, train_loader):
    '''
    Calculates reconstruction accuracy for categorical data
    
    Inputs:
        con_shapes: list of ints corresponding to a number of features each continuous data type have
        con_recon: np.array of VAE model's reconstructions for continuous data
        train_loader: Dataloader of training set
    Returns:
        all_values: list of floats (from 0 to 1) that corresponds to the cosine similarity between input data and reconstructed data
    '''
    
    # Calculate the continuous reconstruction accuracy
    total_shape = 0
    true_recon = []
    cos_values = []
    all_values = []
    for s in con_shapes:
        cor_con = list()
        cos_con = list()
        all_val = list()
        for n in range(0, con_recon.shape[0]):
            con_no_missing = train_loader.dataset.con_all[n,total_shape:(s + total_shape - 1)][train_loader.dataset.con_all[n,total_shape:(s + total_shape - 1)] != 0]
            if len(con_no_missing) <= 1:
                all_val.append(np.nan)
                continue
            con_out_no_missing = con_recon[n,total_shape:(s + total_shape - 1)][train_loader.dataset.con_all[n,total_shape:(s + total_shape - 1)] != 0]
            cor = pearsonr(con_no_missing, con_out_no_missing)[0]
            cor_con.append(cor)

            com = np.vstack([con_no_missing, con_out_no_missing])
            cos = cosine_similarity(com)[0,1]
            cos_con.append(cos)
            all_val.append(cos)

        cor_con = np.array(cor_con)
        cos_con = np.array(cos_con)
        cos_values.append(cos_con)
        all_values.append(np.array(all_val))
        true_recon.append(len(cos_con[cos_con >= 0.9]) / len(cos_con))
        total_shape += s
    return(all_values)

def get_embedding(path, latent):
    '''
    Calculates reconstruction accuracy for categorical data
    
    Inputs:
        path: str of path to results folder
        latent: np.array of VAE latent representation of input dataset
    Returns:
        embedding: np.array of 2D representation of latent space by UMAP
    '''    
    results_folder = path + 'results/'
    
    isExist = os.path.exists(results_folder)
    if not isExist:
         os.makedirs(results_folder)

     # UMAP
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(latent)
    np.save(path + "results/embedding.npy", embedding)
    return(embedding)

def get_feature_data(data_type, feature_of_interest, cat_list,
                           con_list, cat_names, con_names):
    '''
    Returns the data of the selected feature
    
    Inputs:
        data_type (str): 'categorical' or 'continuous' - corresponds to the type of data  
        feature_of_interest (str):  feature name 
        cat_list: list with input data of categorical data type
        con_list: list with input data of continuous data type
        cat_names: np.array of strings of feature names of categorical data
        con_names: np.array of strings of feature names of continuous data
    Returns:
        feature_data: np.array of input data of selected feature
        
    '''
    
    if data_type=='categorical':
        cat_list_integer = [np.argmax(cat, axis=-1) for cat in cat_list]
        np_data_ints = np.concatenate(cat_list_integer, axis=-1)
        headers = cat_names
    elif data_type=='continuous':
        np_data_ints = np.concatenate(con_list, axis=-1)
        headers = con_names
    else:
        raise ValueError("Wrong data type was selected")
     
    feature_data = np_data_ints[:,list(headers).index(feature_of_interest)]
     
    return(feature_data)

def get_pearsonr(feature_of_interest, embedding, 
                 cat_list, con_list, cat_names, con_names):
    '''
    Calculates pearson correlation between input data of feature and UMAP representation 
    
    Inputs:
        feature_of_interest (str): feature name of which pearson correlation is returned
        embedding: np.array of 2D representation of latent space by UMAP
        cat_list: list with input data of categorical data type
        con_list: list with input data of continuous data type
        cat_names: np.array of strings of feature names of categorical data
        con_names: np.array of strings of feature names of continuous data
    Returns:
        pearson_0dim: tuple with pearson correlation and pi value for the 0 dimension of UMAP embedding representation
        pearson_1dim: tuple with pearson correlation and pi value for the 1 dimension of UMAP embedding representation
    '''
    
    if feature_of_interest in cat_names:
        data_type = 'categorical'
    elif feature_of_interest in con_names:
        data_type = 'continuous'
    else:
        raise ValueError("feature_of_interest is not in cat_names or con_names")
    
    feature_data = get_feature_data(data_type, feature_of_interest, 
                                    cat_list, con_list, 
                                    cat_names, con_names)
     
    # Correlate embedding with variable 
    pearson_0dim = pearsonr(embedding[:,0], feature_data)
    pearson_1dim = pearsonr(embedding[:,1], feature_data)
     
    return(pearson_0dim, pearson_1dim)

def get_feature_importance_categorical(model, train_loader, latent, kld_w=1): 
    '''
    Calculates feature importance for categorical data inspired by SHAP - based on how much the latent space changes when setting the values in one hot encoding of the feature to zeroes (corresponds to na value)   
    
    Inputs:
        model: model object 
        train_loader: Dataloader of training set
        latent: np.array of VAE latent representation of input dataset
        kld_w: float of KLD weight

    Returns:
        all_diffs: list: for each categorical feature differences between existing latent space and new latent space (where the feature is set to NA value)   
        all_diffs_cat_np: np.array: for each categorical feature differences between existing latent space and new latent space (where the feature is set to NA value) 
        sum_diffs_cat_np: np.array: for each categorical feature sum of differences of all latent dimensions between existing latent space and new latent space (where the feature is set to NA value) 
        sum_diffs_cat_abs_np: np.array: for each categorical feature sum of absolute differences of all latent dimensions between existing latent space and new latent space (where the feature is set to NA value)  
        total_diffs_cat_np: np.array: for each categorical feature sum among all individuals and of differences of all latent dimensions between existing latent space and new latent space (where the feature is set to NA value) 
    '''
    
    all_diffs = []
    sum_diffs = []
    sum_diffs_abs = []
    total_diffs = []
    loss_cat = []
    pos = 0
    cat_in = train_loader.dataset.cat_all
    cat_shapes = train_loader.dataset.cat_shapes

    for cat_shape in cat_shapes:
        cat_dataset = cat_in[:, pos:(cat_shape[1]*cat_shape[2] + pos)]
        cat_dataset = np.array(cat_dataset.view(cat_in.shape[0], cat_shape[1], cat_shape[2]))
        for feature_index in tqdm(range(cat_shape[1])):

            new_cat = np.copy(cat_dataset)
            new_cat[:,feature_index,:] = 0
            new_cat = new_cat.reshape(new_cat.shape[0], -1)
            input_cat = np.copy(cat_in)
            input_cat[:, pos:(cat_shape[1]*cat_shape[2] + pos)] = new_cat
            input_cat = torch.from_numpy(input_cat)

            dataset = dataloaders.MOVEDataset(input_cat, train_loader.dataset.con_all, 
                                                train_loader.dataset.con_shapes, 
                                                train_loader.dataset.cat_shapes)

            new_loader = DataLoader(dataset, batch_size=1, 
                                        drop_last=False, shuffle=False, 
                                        pin_memory=train_loader.pin_memory) 

            out = model.latent(new_loader, kld_w)

            new_latent_vector = out[0]
            diff = latent-new_latent_vector
            diff_abs = np.abs(latent-new_latent_vector)
            loss_cat.append(out[-1])
            all_diffs.append(diff)
            sum_diffs.append(np.sum(diff, axis = 1))
            sum_diffs_abs.append(np.sum(diff_abs, axis = 1))
            total_diffs.append(np.sum(diff))

    all_diffs_cat_np = np.asarray(all_diffs)
    sum_diffs_cat_np = np.asarray(sum_diffs)
    sum_diffs_cat_abs_np = np.asarray(sum_diffs_abs)
    total_diffs_cat_np = np.asarray(total_diffs)
    
    return(all_diffs, all_diffs_cat_np, sum_diffs_cat_np, sum_diffs_cat_abs_np, total_diffs_cat_np)


def get_feature_importance_continuous(model, train_loader, mask, latent, kld_w=1):
    '''
    Calculates feature importance for continuos data inspired by SHAP - based on how much the latent space changes when setting the values of the feature to zero.   
    
    Inputs:
        model: model object 
        train_loader: Dataloader of training set
        mask: np.array of boaleans, where False values correspond to features that had only NA values.
        latent: np.array of VAE latent representation of input dataset
        kld_w: float of KLD weight

    Returns:  
        all_diffs_con_np: np.array: for each continuous feature differences between existing latent space and new latent space (where the feature is set to NA value) 
        sum_diffs_con_np: np.array: for each continuous feature sum of differences of all latent dimensions between existing latent space and new latent space (where the feature is set to NA value) 
        sum_diffs_con_abs_np: np.array: for each continuous feature sum of absolute differences of all latent dimensions between existing latent space and new latent space (where the feature is set to NA value)  
        total_diffs_con_np: np.array: for each continuous feature sum among all individuals and of differences of all latent dimensions between existing latent space and new latent space (where the feature is set to NA value) 
    '''
# Feature importance continuous

    all_diffs_con = []
    sum_diffs_con = []
    sum_diffs_con_abs = []
    total_diffs_con = []
    loss_con = []
    con_shape = train_loader.dataset.con_all.shape[1]
    for feature_index in tqdm(range(con_shape)):

        new_con = np.array(train_loader.dataset.con_all)
        new_con[:,feature_index] = 0
        new_con = torch.from_numpy(new_con)

        dataset = dataloaders.MOVEDataset(train_loader.dataset.cat_all, new_con,
                                            train_loader.dataset.con_shapes,
                                            train_loader.dataset.cat_shapes)

        new_loader = DataLoader(dataset, batch_size=len(mask), 
                                      drop_last=False, shuffle=False, 
                                      pin_memory=train_loader.pin_memory) #removed num_workers=1,

        out = model.latent(new_loader, kld_w)

        new_latent_vector = out[0]
        loss_con.append(out[-1])
        diff_abs = np.abs(latent-new_latent_vector)
        diff = latent-new_latent_vector
        all_diffs_con.append(diff)
        sum_diffs_con.append(np.sum(diff, axis = 1))
        sum_diffs_con_abs.append(np.sum(diff_abs, axis = 1))
        total_diffs_con.append(np.sum(diff))

    all_diffs_con_np = np.asarray(all_diffs_con)
    sum_diffs_con_np = np.asarray(sum_diffs_con)
    sum_diffs_con_abs_np = np.asarray(sum_diffs_con_abs)
    total_diffs_con_np = np.asarray(total_diffs_con)
    return(all_diffs_con_np, sum_diffs_con_np, sum_diffs_con_abs_np, total_diffs_con_np)


def save_feat_results(path, all_diffs, sum_diffs, sum_diffs_abs, total_diffs, 
                      all_diffs_con, sum_diffs_con, sum_diffs_con_abs, total_diffs_con):
    '''
    Saves feature importance results
    
    Inputs:
        path: str of a pathway where the data is saved   
        all_diffs: np.array: for each categorical feature differences between existing latent space and new latent space (where the feature is set to NA value) 
        sum_diffs: np.array: for each categorical feature sum of differences of all latent dimensions between existing latent space and new latent space (where the feature is set to NA value) 
        sum_diffs_abs: np.array: for each categorical feature sum of absolute differences of all latent dimensions between existing latent space and new latent space (where the feature is set to NA value)  
        total_diffs_cat: np.array: for each categorical feature sum among all individuals and of differences of all latent dimensions between existing latent space and new latent space (where the feature is set to NA value) 
        all_diffs_con: np.array: for each continuous feature differences between existing latent space and new latent space (where the feature is set to NA value) 
        sum_diffs_con: np.array: for each continuous feature sum of differences of all latent dimensions between existing latent space and new latent space (where the feature is set to NA value) 
        sum_diffs_con_abs: np.array: for each continuous feature sum of absolute differences of all latent dimensions between existing latent space and new latent space (where the feature is set to NA value)  
        total_diffs_con: np.array: for each continuous feature sum among all individuals and of differences of all latent dimensions between existing latent space and new latent space (where the feature is set to NA value) 
    '''
    
    # Save results
    all_diffs_both = np.concatenate((all_diffs, all_diffs_con), axis=0)
    sum_diffs_both = np.concatenate((sum_diffs, sum_diffs_con), axis=0)
    sum_diffs_both_abs = np.concatenate((sum_diffs_abs, sum_diffs_con_abs), axis=0)
    total_diffs_both = np.concatenate((total_diffs, total_diffs_con), axis=0)

    np.save(path + "results/all_SHAP_diffs.npy", all_diffs_both)
    np.save(path + "results/sum_diffs.npy", sum_diffs_both)
    np.save(path + "results/sum_diffs_abs.npy", sum_diffs_both_abs)
    np.save(path + "results/total_diffs_final.npy", total_diffs_both)

def get_feat_importance_on_weights(path, model, train_loader, cat_names, con_names):
    '''
    TODO:
    
    Inputs:
        path: str of a pathway where the data is saved
        model: VAE model object 
        train_loader: Dataloader of training set
        cat_names: np.array of strings of feature names of categorical data
        con_names: np.array of strings of feature names of continuous data
    '''
    
    #Based on weights
     
    cat_shapes = train_loader.dataset.cat_shapes
    # get weights
    w = np.array(model.encoderlayers[0].weight.data.to("cpu")) 
    w_sum = np.sum(np.abs(w), axis=0)

    w_sum_con = w_sum[train_loader.dataset.cat_all.shape[1]:]
    w_sum_cat = w_sum[0:train_loader.dataset.cat_all.shape[1]]

    # Get the sum of each input for each categorical one-hot vector
    w_sum_cat_final = []
    pos = 0
    for s in cat_shapes:
        n = s[1] * s[2]
        print(n)
        cat_w_sum_tmp = w_sum_cat[pos:(n + pos)]
        cat_w_sum_tmp = cat_w_sum_tmp.reshape(s[1], s[2])
        sum_d = np.sum(cat_w_sum_tmp, axis=1)
        w_sum_cat_final.extend(sum_d)


    feature_order_cat = np.flip(np.argsort(w_sum_cat_final))
    features_w_cat = cat_names[feature_order_cat]
    # Save the 
    tmp_pd = pd.DataFrame(np.array(w_sum_cat_final)[feature_order_cat], index=features_w_cat)
    tmp_pd.T.to_csv(path + "results/importance_w_cat.txt")

    feature_order = np.flip(np.argsort(w_sum_con))
    features_w_con = con_names[feature_order]

    tmp_pd = pd.DataFrame(w_sum_con[feature_order], index=features_w_con)
    tmp_pd.T.to_csv(path + "results/importance_w_con.txt")
     
def cal_reconstruction_change(recon_results, repeats):
    '''
    Calculates reconstruction change across repeats.
    
    Inputs: 
        recon_results (dict): {latents: {repeat: {drug: np.array of changes in continuous data when label of drug is changed}}}
        repeats (int): number of repeats
    Returns:
        recon_average (dict): {latents: {drug: np.array of mean changes among different repeats in continuous data when label of drug is changed}}
    '''
    
    recon_average = dict()
    for latent in recon_results.keys():
        average = defaultdict(dict)
        for repeat in range(len(recon_results[latent])):
            for drug in range(len(recon_results[latent][repeat])):
                tmp_recon = recon_results[latent][repeat][drug]
                if drug in average:
                    average[drug] = np.add(average[drug], tmp_recon)
                else:
                    average[drug] = tmp_recon
        a = {k: (v / repeats) for k, v in average.items()}
        recon_average[latent] = a
    return(recon_average)


def overlapping_hits(nLatents, cor_results, repeats, con_names, drug): 
    '''
    Identifies overlapping hits in the repeats on the same latent space size
    
    Inputs:
        nLatents: list of ints with size of latent space
        cor_results: TODO
        repeats (int): number of repeats
        con_names: np.array of strings of feature names of continuous data
        drug: np.array of input data whose features' data are changed to test their effects in the pipeline
    Returns:
        sig_hits: TODO
        median_p_val: TODO
    '''
    
    
    sig_hits = defaultdict(dict)
    overlaps_d = defaultdict(list)
    counts = list()

    new_list = nLatents[::-1]

    median_p_val = defaultdict(dict)
    for l in range(len(new_list)):
        for d in range(cor_results[0][new_list[l]].shape[0]):
            hits_tmp = list()
            p_cors = defaultdict(list)
            for repeat in range(repeats):

                ns = con_names[cor_results[repeat][new_list[l]][d,:] <= 0.05]

                p_cor = cor_results[repeat][new_list[l]][d,:]
                p_cor = p_cor[p_cor <= 0.05]
                for i,ns_t in enumerate(ns):
                    p_cors[ns_t].append(p_cor[i])

                hits_tmp.extend(ns)

            overlap_tmp = [hits_tmp.count(x) for x in np.unique(hits_tmp)]
            overlap = np.array(np.unique(hits_tmp))[np.array(overlap_tmp) >= 5]
            m_p = []
            for o_t in overlap:
                m_p.append(np.median(p_cors[o_t]))

            sig_hits[d][new_list[l]] = overlap
            median_p_val[d][new_list[l]] = m_p
    return(sig_hits, median_p_val)

def identify_high_supported_hits(sig_hits, drug_h, version, path): 
    '''
    Get significant hits found in multiple sizes of the latent space
    
    Inputs:
        sig_hits: TODO
        drug_h: np.array of strings of feature names data type whose data are changed to test their effects in the pipeline
        version: str of subfolder name where the results will be saved
        path: str of folder name where the results will be saved
    Returns:
        all_hits: TODO
        collected_overlap: TODO
    '''
    
    result = dict()
    collected_overlap = defaultdict(list)
    all_hits = list()
    for d in sig_hits:
        result[drug_h[d]] = defaultdict(list)
        all_h = []
        for l in sig_hits[d]:
            all_h.extend(sig_hits[d][l])

        for x in set(all_h):
            if all_h.count(x) >=3:
                result[drug_h[d]]['high'].append(x)
            elif all_h.count(x) == 2:
                result[drug_h[d]]['medium'].append(x)
            else:
                result[drug_h[d]]['low'].append(x)

            if all_h.count(x) >= 3:
                collected_overlap[drug_h[d]].append(x)
                all_hits.append(x)
    
    # Save result
    np.save(path + "results/results_confidence_" + version + ".npy", result)

    return(all_hits, collected_overlap)


def report_values(path, sig_hits, median_p_val, drug_h, all_hits, collected_overlap, con_names): 
    '''
    Saves the pi values of results of overlapping_hits() and  identify_high_supported_hits() functions
    
    Inputs:
        path: str of folder name where the results will be saved
        sig_hits: TODO
        median_p_val: TODO
        drug_h: np.array of strings of feature names data type whose data are changed to test their effects in the pipeline
        all_hits: TODO
        con_names: np.array of strings of feature names of continuous data
    '''
    
    
    results_folder = path + 'results/sig_ci_files'
    isExist = os.path.exists(results_folder)
    if not isExist:
        os.makedirs(results_folder)

    p_vals = list()
    for d in sig_hits:
        p_vals_col = []
        p_vals_tmp = defaultdict(list)
        for l in sig_hits[d]:
            name_s = sig_hits[d][l]
            ps = median_p_val[d][l]
            for i,ns_1 in enumerate(name_s):
                if ns_1 in collected_overlap[drug_h[d]]:
                    p_vals_tmp[ns_1].append(ps[i])

        m_p_vals_tmp = dict()
        for ns_2 in p_vals_tmp:
            m_p_vals_tmp[ns_2] = np.median(p_vals_tmp[ns_2])

        for a_h in all_hits:
            if a_h in m_p_vals_tmp:
                p_vals_col.append(m_p_vals_tmp[a_h])
            else:
                p_vals_col.append('ns')

        p_vals.append(p_vals_col)

    p_vals_df = pd.DataFrame(p_vals, index=drug_h, columns=all_hits)

    # Save the files for each continuous dataset
    for i,al_con in enumerate(con_names):
        sig_drug_names = np.intersect1d(all_hits, al_con)
        df_tmp = p_vals_df.loc[:, sig_drug_names]
        df_tmp.T.to_csv(path + "results/sig_ci_files/" + con_names[i] + "_p_vals.txt", sep = "\t")


def get_change_in_reconstruction(recon_average, groups, drug, drug_h, con_names, collected_overlap, sig_hits, con_all, version, path, types): 
    '''
    TODO
    
    Inputs:
        recon_average: TODO
        groups: TODO:
        drug: np.array of input data whose features' data are changed to test their effects in the pipeline
        drug_h: np.array of strings of feature names data type whose data are changed to test their effects in the pipeline
        con_names: np.array of strings of feature names of continuous data
        collected_overlap: TODO
        sig_hits: TODO
        con_all: TODO
        version: str of subfolder name where the results will be saved
        path: str of folder name where the results will be saved
        types: TODO
    Returns:
        recon_average_corr_new_all: TODO 
        recon_average_corr_all_indi_new: TODO
    '''
    
    recon_average_corr_all = dict()
    counts_average_all = dict()
    recon_average_corr_all_indi = dict()
    counts_indi = dict()

    for l in recon_average.keys():
        for d in recon_average[l].keys():
            print(d)
            tmp_recon = np.copy(recon_average[l][d])
            gr = groups[d]
            g = [not (np.all(a_s == types[0]) or (np.all(a_s == [0,0]))) for a_s in drug[gr,d,:]]
            tmp_recon = tmp_recon[g]

            if d not in recon_average_corr_all:
                recon_average_corr_all[d] = [0] * len(con_names)
                counts_average_all[d] = [0] * len(con_names)
                recon_average_corr_all_indi[d] = [[0] * tmp_recon.shape[1]] * tmp_recon.shape[0]
                counts_indi[d] = [[0] * tmp_recon.shape[1]] * tmp_recon.shape[0]

            for f in tqdm(range(tmp_recon.shape[1])):
                if con_names[f] in collected_overlap[drug_h[d]]:
                    if con_names[f] in sig_hits[d][l]:
                        tmp_vals = con_all[groups[d],f]
                        tmp_vals = tmp_vals[g]
                        avg_tmp = np.mean(tmp_recon[tmp_vals != 0,f])
                        recon_average_corr_all[d][f] += avg_tmp
                        counts_average_all[d][f] += 1

                        for indi in range(tmp_recon.shape[0]):
                            if tmp_vals[indi] != 0:
                                recon_average_corr_all_indi[d][indi][f] += tmp_recon[indi,f]
                                counts_indi[d][indi][f] += 1
                else:
                    tmp_vals = con_all[groups[d],f]
                    tmp_vals = tmp_vals[g]
                    avg_tmp = np.mean(tmp_recon[tmp_vals != 0,f])
                    recon_average_corr_all[d][f] += avg_tmp
                    counts_average_all[d][f] += 1

                    for indi in range(tmp_recon.shape[0]):
                        if tmp_vals[indi] != 0:
                            recon_average_corr_all_indi[d][indi][f] += tmp_recon[indi,f]
                            counts_indi[d][indi][f] += 1

    recon_average_corr_new_all = list()
    recon_average_corr_all_indi_new = dict()
    for d in recon_average_corr_all.keys():
        print(d)
        counts_tmp = np.array(counts_average_all[d])
        tmp_l = np.array(recon_average_corr_all[d])[counts_tmp != 0]
        included_names = con_names[counts_tmp != 0]
        counts_tmp = counts_tmp[counts_tmp != 0]
        recon_average_corr_new_all.append(tmp_l/counts_tmp)

        tmp_recon_average_corr_all_indi_new = list()
        for f in range(len(recon_average_corr_all_indi[d])):
            tmp_recon_average_corr_all_indi_new.append(np.array(recon_average_corr_all_indi[d][f]) / np.array(counts_indi[d][f]))

        recon_average_corr_all_indi_new[d] = np.transpose(np.array(tmp_recon_average_corr_all_indi_new))

    recon_average_corr_new_all = np.array(recon_average_corr_new_all)

     # Save recon results
    np.save(path + "results/results_confidence_recon_all_" + version + ".npy", recon_average_corr_new_all)
    np.save(path + "results/results_confidence_recon_all_indi_" + version + ".npy", recon_average_corr_all_indi_new)

    return(recon_average_corr_new_all, recon_average_corr_all_indi_new)


def write_omics_results(path, up_down_list, collected_overlap, recon_average_corr_new_all, headers_all, con_types, drug_h, con_names): 
    '''
    TODO:
    
    Inputs:
        path: str of folder name where the results will be saved
        up_down_list: list of strs that correspond to data types whose upregulated or downregulated data points would be saved 
        collected_overlap: TODO
        recon_average_corr_new_all: TODO
        headers_all: np.array of strings of feature names of all data
        con_types: list of strings of continuous data type names
        drug_h: np.array of strings of feature names data type whose data are changed to test their effects in the pipeline
        con_names: np.array of strings of feature names of continuous data
    
    '''
 
    for i in range(len(con_types)):
        for d in collected_overlap:
            n = np.intersect1d(collected_overlap[d], headers_all[i])

            with open(path + f"results/{con_types[i]}_" + d.replace(" ", "_") + ".txt", "w") as o:
                o.write("\n".join(n))  

            if con_types[i] in up_down_list:

                vals = recon_average_corr_new_all[list(drug_h).index(d),np.where(np.isin(con_names,n))[0]]
                up = n[vals > 0]
                down = n[vals < 0]
                with open(path + f"results/{con_types[i]}_up_" + d.replace(" ", "_") + ".txt", "w") as o:
                    o.write("\n".join(up))

                with open(path + f"results/{con_types[i]}_down_" + d.replace(" ", "_")  + ".txt", "w") as o:
                    o.write("\n".join(down))

                        
def make_files(collected_overlap, groups, con_all, path, recon_average_corr_all_indi_new, 
               con_names, con_dataset_names, drug_h, drug, all_hits, types, version = "v1"):
    '''
    TODO:
    
    Inputs:
        collected_overlap: TODO
        groups: TODO
        con_all: np.array of data of continuous data type
        path: str of folder name where the results will be saved
        recon_average_corr_all_indi_new: TODO
        con_names: np.array of strings of feature names of continuous data
        con_dataset_names: list of strings with the names of continuous data type
        drug: np.array of input data whose feature data are changed to test their effects in the pipeline
        drug_h: np.array of strings of feature names data type whose data are changed to test their effects in the pipeline
        all_hits: TODO
        types (list of lists): TODO
        version: str: a subdirectory where data will be saved
    
    '''
    
    
#     all_db_names = [item for sublist in con_names for item in sublist]
    all_db_names = con_names
    ci_dict = {}
    for i,n in enumerate(con_dataset_names):
        ci_collected = []
        sig_drug_names = np.intersect1d(all_hits, n)
        for d in drug_h:
            f = drug_h.index(d)
            recon_data_d = recon_average_corr_all_indi_new[f]
            
            gr = groups[f]
            
            sig_data = recon_data_d[:,np.where(np.isin(all_db_names,sig_drug_names))[0]]
            g = [not (np.all(a_s == types[0]) or (np.all(a_s == [0,0]))) for a_s in drug[gr,f,:]]
            con_tmp = con_all[gr]
            con_tmp = con_tmp[g]
            sig_names_sort = np.array(all_db_names)[np.where(np.isin(all_db_names,sig_drug_names))]
            sig_part_df = pd.DataFrame(sig_data, columns = sig_names_sort)
            sig_part_df = sig_part_df.T
            sig_part_df[np.isnan(sig_part_df)] = 0
            
            ci_all = []
            for j in sig_part_df.index:
                data_vals = np.array(sig_part_df.loc[j,:])
                con_vals = con_tmp[:,all_db_names.index(j)]
                data_vals = data_vals[con_vals != 0]
                ci = stats.t.interval(0.95, len(data_vals)-1, loc=np.nanmean(data_vals), scale=stats.sem(data_vals))
                ci_all.append("%.4f [%.4f, %.4f]"%(np.mean(data_vals), ci[0], ci[1]))
            
            ci_collected.append(ci_all)
        
        ci_collected_df = pd.DataFrame(ci_collected, index = drug_h, columns=sig_names_sort)
        ci_collected_df.T.to_csv(path + "results/" + con_dataset_names[i] + "_ci_sig_" + version +  ".txt", sep = "\t")
        
        
def get_inter_drug_variation(con_names, drug_h, recon_average_corr_all_indi_new, 
                             groups, collected_overlap, drug, con_all, path, types):
    '''
    TODO:
    
    Inputs:
        con_names: np.array of strings of feature names of continuous data
        drug_h: np.array of strings of feature names data type whose data are changed to test their effects in the pipeline
        recon_average_corr_all_indi_new: TODO 
        groups: TODO
        collected_overlap: TODO 
        drug: np.array of input data whose feature data are changed to test their effects in the pipeline
        con_all:  np.array of data of continuous data type
        path: str of folder name where the results will be saved
        types (list of lists): TODO
        
    Returns:
        df_indi_var: TODO
    '''
    
    # Inter drug variation 
#     all_db_names = [item for sublist in con_names for item in sublist]
    all_db_names = con_names
    inter_drug_variance = []
    inter_drug_std = []
    for d in drug_h:
        f = drug_h.index(d)
        recon_data_d = recon_average_corr_all_indi_new[f]
        gr = groups[f]
        sig_drug_names = collected_overlap[d]
        sig_data = recon_data_d[:,np.where(np.isin(all_db_names,sig_drug_names))[0]]
        g = [not (np.all(a_s == types[0]) or (np.all(a_s == [0,0]))) for a_s in drug[gr,f,:]]
        con_tmp = con_all[gr]
        con_tmp = con_tmp[g]
        sig_part_df = pd.DataFrame(sig_data, columns = sig_drug_names)
        sig_part_df = sig_part_df.T
        sig_part_df[np.isnan(sig_part_df)] = np.nan
        inter_drug_variance.append(np.nanvar(sig_part_df))
        inter_drug_std.append(np.nanstd(sig_part_df))

    df_indi_var = pd.DataFrame(inter_drug_variance, index=drug_h)
    return(df_indi_var)


def get_drug_similar_each_omics(con_names, con_dataset_names, all_hits, recon_average_corr_new_all, drug_h, version, path):
    '''
    TODO:
    
    Inputs:
        con_names: np.array of strings of feature names of continuous data
        con_dataset_names: list of strings with the names of continuous data type
        all_hits: TODO
        recon_average_corr_new_all: TODO
        drug_h: np.array of strings of feature names data type whose data are changed to test their effects in the pipeline
        version: str: a subdirectory where data will be saved
        path: str of folder name where the results will be saved
    '''
    
    con_dataset_names_v1 = con_dataset_names
    i = 0
    for n in con_names:
        tmp = np.intersect1d(all_hits, n)
        if len(tmp) == 0:
             continue

        sig_data = recon_average_corr_new_all[:,np.where(np.isin(all_hits,n))[0]]
        sim = cosine_similarity(sig_data)
        corr = pd.DataFrame(sim, columns = drug_h, index = drug_h)
        sig_data = pd.DataFrame(corr, columns = drug_h, index = drug_h)
        g = sns.clustermap(sig_data, cmap=cmap, center=0, xticklabels = True,
                           yticklabels = True, metric='correlation',
                           linewidths=0, row_cluster=True, col_cluster=True, figsize=(10,10))

        g.fig.suptitle(con_dataset_names_v1[i])
        g.fig.subplots_adjust(top=0.9)
        plt.savefig(path + "results/" + con_dataset_names[i] + "_heatmap_" + version + "_all.pdf", format = 'pdf', dpi = 800)
        i += 1

    plt.close('all')
