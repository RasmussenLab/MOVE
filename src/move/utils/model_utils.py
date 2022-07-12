import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from scipy.stats.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import itertools 
import tqdm    
from scipy import stats
from statsmodels.stats.multitest import multipletests

from move.utils import dataloaders
# Funtions for calculations

def get_latent(best_model, train_loader, test_loader, kld_w):
    # Get training set results
    train_test_loader = DataLoader(dataset=train_loader.dataset, batch_size=1, drop_last=False, shuffle=False) #, num_workers=1, pin_memory=train_loader.pin_memory

    latent, latent_var, cat_recon, cat_class, con_recon, loss, likelihood = best_model.latent(train_test_loader, kld_w)
    con_recon = np.array(con_recon)

    con_recon = torch.from_numpy(con_recon)
    
    # Get test set results
    test_loader = DataLoader(dataset=test_loader.dataset, batch_size=1, drop_last=False, shuffle=False) #, num_workers=1, pin_memory=test_loader.pin_memory
    latent_test, latent_var_test, cat_recon_test, cat_class_test, con_recon_test, loss_test, likelihood_test = best_model.latent(test_loader, kld_w)
    con_recon_test = np.array(con_recon_test)
    con_recon_test = torch.from_numpy(con_recon_test)
    
    return con_recon, train_test_loader, latent, latent_var, cat_recon, cat_class, loss, likelihood, latent_test, latent_var_test, cat_recon_test, cat_class_test, con_recon_test, loss_test, likelihood_test

def cal_recon(cat_shapes, cat_recon, cat_class, train_loader, con_recon, con_shapes, cat_recon_test, cat_class_test, test_loader, con_recon_test):
    cat_true_recon = cal_cat_recon(cat_shapes, cat_recon, cat_class)
    true_recon = cal_con_recon(train_loader, con_recon, con_shapes)
    cat_true_recon_test = cal_cat_recon(cat_shapes, cat_recon_test, cat_class_test)
    true_recon_test = cal_con_recon(test_loader, con_recon_test, con_shapes)
    
    return cat_true_recon,true_recon,cat_true_recon_test,true_recon_test


# def save_input(combi,cat_true_recon,true_recon,latent,con_recon,cat_recon,loss,likelihood,best_epoch,cat_true_recon_test,true_recon_test,latent_test,con_recon_test,cat_recon_test,loss_test,likelihood_test):
  

#    recon_acc[combi].append(cat_true_recon + true_recon)
#    latents[combi].append(latent)
#    con_recons[combi].append(con_recon)
#    cat_recons[combi].append(cat_recon)
#    loss_train[combi].append(loss)
#    likelihoods[combi].append(likelihood)
#    best_epochs[combi].append(best_epoch)
  
#    recon_acc_tests[combi].append(cat_true_recon_test + true_recon_test)
#    latents_tests[combi].append(latent_test)
#    con_recons_tests[combi].append(con_recon_test)
#    cat_recons_tests[combi].append(cat_recon_test)
#    loss_tests[combi].append(loss_test)
#    likelihood_tests[combi].append(likelihood_test)
#    return(recon_acc, latents, con_recons, cat_recons, loss_train, likelihoods, best_epochs, recon_acc_tests,\
#             latents_tests, con_recons_tests, cat_recons_tests, loss_tests, likelihood_tests)

def cal_cat_recon(cat_shapes, cat_recon, cat_class):
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
    return cat_true_recon

def cal_con_recon(train_loader, con_recon, con_shapes):
    total_shape = 0
    true_recon = []
    cos_values = []
    all_values = []
    for s in con_shapes:
        cor_con = list()
        cos_con = list()
        all_val =list()
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
        true_recon.append(len(cos_con[cos_con >= 0.7]) / len(cos_con))
        total_shape += s
        
    return true_recon

# Functions for calculations

def get_baseline(model, train_loader, con_recon, repeat=25, kld_w=0):
    recon_diff_baseline = list()
    for r in range(repeat):
        latent_new, latent_var_new, cat_recon_new, cat_class_new, con_recon_new, test_loss_new, likelihood_new = model.latent(train_loader, kld_w)
    
        recon_diff = con_recon_new - np.array(con_recon) 
        recon_diff_baseline.append(recon_diff)
    
    matrix = np.array(recon_diff_baseline)
    baseline_mean = np.mean(matrix, axis= 0)
    baseline_mean = np.where(train_loader.dataset.con_all == 0, np.NaN , baseline_mean)
    return baseline_mean

def change_drug_atc(train_loader, trans_table, con_recon, drug, types = [[1,0]], data_start=1557):
    types = np.array(types)
    
    recon_diff_con_none = dict()
    none_groups = dict()
    data_shapes = drug.shape
    data_end = train_loader.dataset.cat_all.shape[1]
    for feature_index in range(data_shapes[1]):
        atc = trans_table[drug_h[feature_index]]
        same_atc = [k for k, v in trans_table.items() if (v == atc and k != drug_h[feature_index])]
        same = 0
        drug_indexes = []
        if len(same_atc) > 0:
            same = 1
            for i in same_atc:
                drug_indexes.append(drug_h.index(i))
        
        data = np.array(train_loader.dataset.cat_all)
        
        for t in types:
            tmp_data = np.copy(data[:,data_start:data_end])
            tmp_data = tmp_data.reshape(tmp_data.shape[0], data_shapes[1], data_shapes[2])
            tmp_data[:,feature_index,:] = t
            
            same_group = [True] * data.shape[0]
            none_group = [True] * data.shape[0]
            if same != 0:
                for i in drug_indexes:
                    same_group_tmp = [(np.all(a_s == types[0])) for a_s in drug[:,i,:]]
                    none_group_tmp = [not (np.all(a_s == types[0]) or (np.all(a_s == [0,0]))) for a_s in drug[:,i,:]]
                    
                    same_group = [all(tup) for tup in zip(same_group, same_group_tmp)]
                    none_group = [all(tup) for tup in zip(none_group, none_group_tmp)]
            
            tmp_data = tmp_data.reshape(tmp_data.shape[0], -1)
            new_data = np.copy(data)
            new_data[:,data_start:data_end] = tmp_data
            new_data = torch.from_numpy(new_data)
            
            none_groups[feature_index] = none_group
            new_data = new_data[none_group,:]
            dataset = dataloaders.MOVEDataset(new_data, train_loader.dataset.con_all[none_group,:], train_loader.dataset.con_shapes, train_loader.dataset.cat_shapes)
            
            new_loader = DataLoader(dataset, batch_size=train_loader.batch_size, drop_last=False,
                                    shuffle=False, num_workers=1, pin_memory=train_loader.pin_memory)
            
            latent_new, latent_var_new, cat_recon_new, cat_class_new, con_recon_new, test_loss_new, likelihood_new = model.latent(new_loader, kld_w)
            
            recon_diff = (con_recon_new - np.array(con_recon[none_group,:]))
            recon_diff_con_none[feature_index] = recon_diff
    
    return recon_diff_con_none,none_groups


def change_drug(model, train_loader, con_recon, drug, start, end, kld_w, types = [[1,0]]):
    types = np.array(types)
    
    recon_diff_con_none = dict()
    none_groups = dict()
    data_shapes = drug.shape
    data_start = start
    data_end = end #train_loader.dataset.cat_all.shape[1] # todo: Change to the end of the category 
    for feature_index in range(data_shapes[1]):
        data = np.array(train_loader.dataset.cat_all)
        for t in types:
            tmp_data = np.copy(data[:,data_start:data_end])
            tmp_data = tmp_data.reshape(tmp_data.shape[0], data_shapes[1], data_shapes[2])
            tmp_data[:,feature_index,:] = t
            
            none_group = [True] * data.shape[0]
            
            tmp_data = tmp_data.reshape(tmp_data.shape[0], -1)
            new_data = np.copy(data)
            new_data[:,data_start:data_end] = tmp_data
            new_data = torch.from_numpy(new_data)
            
            none_groups[feature_index] = none_group
            dataset = dataloaders.MOVEDataset(new_data, train_loader.dataset.con_all, train_loader.dataset.con_shapes, train_loader.dataset.cat_shapes)
            
            new_loader = DataLoader(dataset, batch_size=train_loader.batch_size, drop_last=False,
                                    shuffle=False,  pin_memory=train_loader.pin_memory) #num_workers=1,
            
            latent_new, latent_var_new, cat_recon_new, cat_class_new, con_recon_new, test_loss_new, likelihood_new = model.latent(new_loader, kld_w)
            
            recon_diff = (con_recon_new - np.array(con_recon))
            recon_diff_con_none[feature_index] = recon_diff
    
    return recon_diff_con_none,none_groups


def cal_sig_hits(recon_diff_con_none, none_groups, drug, baseline_mean, con_all, types = [[1,0]]):
    data_shapes = drug.shape
    none_avg = list()
    none_stats = list()
    for f in range(0,data_shapes[1]):
        if f in none_groups:
            tmp = np.copy(recon_diff_con_none[f])
            tmp_data = con_all[none_groups[f],:]
            tmp_baseline = baseline_mean[none_groups[f],:]
            g = [not (np.all(a_s == types[0]) or (np.all(a_s == [0,0]))) for a_s in drug[none_groups[f],f,:]]
            tmp = np.where(tmp_data[g,:] == 0,np.NaN, tmp[g,:])
            tmp_abs = np.abs(tmp)
            stat = stats.ttest_rel(tmp, tmp_baseline[g,:], axis=0, nan_policy="omit")
            
            if np.all(np.isnan(stat[1])):
                p_stat = np.zeros((con_all.shape[1]))
                p_stat[:] = np.nan
            else:
                p_stat = stat[1]
            
            none_stats.append(p_stat)
            avg = np.nanmean(tmp, axis = 0)
            avg[np.isnan(avg)] = 0
            none_avg.append(avg)
        else:
            tmp = np.zeros((tmp_data.shape[1]))
            tmp[:] = np.nan
            none_avg.append(tmp)
            none_stats.append(tmp)
    
    none_avg = np.array(none_avg)
    none_stats = np.array(none_stats)
    
    return none_stats

def correction_new(results):
    new_results = defaultdict(dict)
    for l in results:
        for r in range(len(results[l])):
            corrected = np.zeros((results[l][r].shape[0],results[l][r].shape[1]))
            for d in range(results[l][r].shape[0]):
                stats_cor = multipletests(results[l][r][d,:], method = "bonferroni")[1]
                corrected[d,:] = stats_cor
            
            new_results[r][l] = corrected
    
    return new_results

# def get_start_end_positions(cat_list, data_dict):
  
#    n_cat = 0
#    cat_shapes = list()
#    cat_all = []
#    first = 0
#    for i in range(len(data_dict['categorical_data_files'])):
#      cat_d = cat_list[i]
     
#      cat_shapes.append(cat_d.shape)
#      cat_input = cat_d.reshape(cat_d.shape[0], -1)
     
     
#      if first == 0:
#         cat_all = cat_input
#         del cat_input
#         first = 1
     
#         if data_dict['data_of_interest'] == data_dict['categorical_data_files'][i]:
#            start = 0
#            end = cat_all.shape[-1]
          
#      else:
#         cat_all = np.concatenate((cat_all, cat_input), axis=1)
        
#         if data_dict['data_of_interest'] == data_dict['categorical_data_files'][i]:
#            start = cat_all.shape[-1] - cat_input.shape[-1]
#            end = cat_all.shape[-1]
     
#    # Make mask for patients with no measurments
#    catsum = cat_all.sum(axis=1)
#    mask = catsum > 5
#    del catsum
#    return start, end



def get_start_end_positions(cat_list, categorical_names, data_of_interest):
  
    n_cat = 0
    cat_shapes = list()
    cat_all = []
    first = 0
    for i in range(len(categorical_names)):
        cat_d = cat_list[i]
     
        cat_shapes.append(cat_d.shape)
        cat_input = cat_d.reshape(cat_d.shape[0], -1)
     
     
        if first == 0:
            cat_all = cat_input
            del cat_input
            first = 1
     
            if data_of_interest == categorical_names[i]:
                start = 0
                end = cat_all.shape[-1]
          
        else: 
            cat_all = np.concatenate((cat_all, cat_input), axis=1)
        
            if data_of_interest == categorical_names[i]:
                start = cat_all.shape[-1] - cat_input.shape[-1]
                end = cat_all.shape[-1]
     
    # Make mask for patients with no measurments
    catsum = cat_all.sum(axis=1)
    mask = catsum > 5
    del catsum
    return start, end
