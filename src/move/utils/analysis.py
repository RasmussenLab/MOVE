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
     
    return(feature_data, headers)

def get_pearsonr(feature_of_interest, embedding, 
                 cat_list, con_list, cat_names, con_names):
     
    if feature_of_interest in cat_names:
        data_type = 'categorical'
    elif feature_of_interest in con_names:
        data_type = 'continuous'
    else:
        raise ValueError("feature_of_interest is not in cat_names or con_names")
    
    feature_data, _ = get_feature_data(data_type, feature_of_interest, 
                                                   cat_list, con_list, 
                                                   cat_names, con_names)
     
    # Correlate embedding with variable 
    pearson_0dim = pearsonr(embedding[:,0], feature_data)
    pearson_1dim = pearsonr(embedding[:,1], feature_data)
     
    return(pearson_0dim, pearson_1dim)

def get_feature_importance_categorical(model, train_loader, latent, kld_w=1): #Which kld_w and train_loader # should not matter equal to 1

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
                                        pin_memory=train_loader.pin_memory) # removed num_workers=1,

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
    recon_average = dict()
    for l in recon_results.keys():
        average = defaultdict(dict)
        for r in range(len(recon_results[l])):
            for d in range(len(recon_results[l][r])):
                tmp_recon = recon_results[l][r][d]
                if d in average:
                    average[d] = np.add(average[d], tmp_recon)
                else:
                    average[d] = tmp_recon
        a = {k: (v / repeats) for k, v in average.items()}
        recon_average[l] = a
    return(recon_average)


def overlapping_hits(nLatents, cor_results, repeats, con_names, drug): 
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




def report_values(path, sig_hits, median_p_val, drug_h, all_hits, con_names): 

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


def write_omics_results(path, up_down_list, collected_overlap, recon_average_corr_new_all, headers_all, con_types, data_of_interest): 
 
    for i in range(len(con_types)):
        if con_types[i] != data_of_interest:
            for d in collected_overlap:
                n = np.intersect1d(collected_overlap[d], headers_all[i])
                
                with open(path + f"results/{con_types[i]}_" + d.replace(" ", "_") + ".txt", "w") as o:
                    o.write("\n".join(n))  
                
                if con_types[i] in up_down_list:
                     
                    vals = recon_average_corr_new_all[list(headers_all[i]).index(d),np.where(np.isin(con_names,n))[0]]
                    up = n[vals > 0]
                    down = n[vals < 0]
                    with open(path + f"results/{con_types[i]}_up_" + d.replace(" ", "_") + ".txt", "w") as o:
                        o.write("\n".join(up))

                    with open(path + f"results/{con_types[i]}_down_" + d.replace(" ", "_")  + ".txt", "w") as o:
                        o.write("\n".join(down))
                        
                        
def make_files(collected_overlap, groups, con_all, path, recon_average_corr_all_indi_new, 
               con_names, con_dataset_names, drug_h, drug, all_hits, types, version = "v1"):
    
    all_db_names = [item for sublist in con_names for item in sublist]
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
    # Inter drug variation 
    all_db_names = [item for sublist in con_names for item in sublist]
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


    
def get_best_epoch(results_df):
    
    #Rounding best_epoch to closest 10 (just in case if best_epoch is less than 5 - to closest number)
    round_epoch = lambda x : round(x, 1) if (x >= 5) else round(x, 0)   
    best_epoch = results_df['best_epochs'].mean()
    best_epoch = int(round_epoch(best_epoch))
    
    return(best_epoch)


def get_significant_param_values(results_df: pd.DataFrame, 
                                 list_of_params: list, 
                                 metric_name: str = 'likelihood_test', 
                                 stat_test_name: str = 'ttest_rel'):
    
    params_dict_to_remove = defaultdict(list)
    stat_test = getattr(stats, stat_test_name) 
        
    # Iterating through each of the hypeparameter
    for col in list_of_params:

        # Getting each value of the hyperparameter
        unique_values = results_df[col].unique()


        # finding which pairs of parameter values do not have significant differences
        insignif_pairs_list = []
        for i in range(len(unique_values)):
            for j in range(i+1, len(unique_values)):
                
                # Getting the p-value of statistical results by comparing hyperparameter values sets where only the tested hyperparameter is different
                _, pvalue = stat_test(
                    results_df.loc[results_df[col] == unique_values[i], metric_name],
                    results_df.loc[results_df[col] == unique_values[j], metric_name])

                if pvalue < 0.05:
                    insignif_pairs_list.append(tuple([unique_values[j], 
                                                      unique_values[i]]))

        # Removing the parameter that did not show significant differences (in the order from the most insignificant pairs)
        # strategy: remove the unique value with most insignificant results first, repeat until none is left
        while insignif_pairs_list:
            insignif_values_dict = Counter(value for values_pair in insignif_pairs_list for value in values_pair)
            value_max_occurs = max(insignif_values_dict, key=insignif_values_dict.get)
            insignif_pairs_list = [x for x in insignif_pairs_list if value_max_occurs not in x]
            if params_dict_to_remove.get(col) is None:
                params_dict_to_remove[col] = []
            params_dict_to_remove[col].append(value_max_occurs.item())
                

    # Removing the selected parameters to remove from the dataframe
    results_df_rm_nonsignifs = pd.DataFrame(results_df)
    for key, value_list in params_dict_to_remove.items():
        results_df_rm_nonsignifs = results_df_rm_nonsignifs[results_df_rm_nonsignifs[key].isin(value_list)]
    return(results_df_rm_nonsignifs, params_dict_to_remove)

def get_sort_list(results_df):
    
    #Gets mean values of test accuracy reconstruction
    results_df['recon_acc_test_mean'] = results_df['recon_acc_test'].map(lambda x: x.mean())
    
    # Sort values by reconstruction accuracy
    results_df = results_df.sort_values('recon_acc_test_mean', ascending=False)
    return (results_df)


def get_length(hyperpars_vals_dict, hyperpar_name):
    lengths = 1
    for key in hyperpars_vals_dict:  
        length = len(hyperpars_vals_dict[key])
        # Adding +1 since we want to check if adding a parameter it overreaches
        if key == hyperpar_name:
            length+=1
        lengths *= length

    return(lengths)

def get_best_params(results_df_sorted, n_combos_opt, hyperpars_names):

    hyperpars_vals_dict = defaultdict(list)
    for index, row in results_df_sorted.iterrows():
        
        for hyperpar_name in hyperpars_names:
            if row[hyperpar_name] not in hyperpars_vals_dict[hyperpar_name]:
                
                length = get_length(hyperpars_vals_dict, hyperpar_name)
                if length <= n_combos_opt:
                    hyperpars_vals_dict[hyperpar_name].append(row[hyperpar_name])
                else:
                    break
    hyperpars_vals_dict = dict(hyperpars_vals_dict)
    return(hyperpars_vals_dict)

def make_and_save_best_reconstruct_params(results_df, hyperparams_names, max_param_combos_to_save):
    
    # Getting the best number of epochs used in further trainings
    best_epoch = get_best_epoch(results_df)
    
    print('Starting calculating the best hyperparameter values for further optimization') 
    
    # Removing insignificant hyperparameter values
    results_df_rm_insignifs, params_dict_to_remove = get_significant_param_values(results_df, hyperparams_names)
    
    #Printing the removed parameter values
    print(f'\nRemoved insignificant parameters:\n {OmegaConf.to_yaml(dict(params_dict_to_remove))}')
    
    # Getting up to n combinations of parameters that will be used further to optimize stability
    results_df_sorted = get_sort_list(results_df_rm_insignifs)
    best_hyperpars_vals_dict = get_best_params(results_df_sorted, max_param_combos_to_save, hyperparams_names)
    best_hyperpars_vals_dict['tuned_num_epochs'] = best_epoch
    
    # Saving the best hyperparameter values (it will overwrite the file if it already exists)
    with open('tuning_stability.yaml', "w") as f:
        OmegaConf.save(OmegaConf.create(dict(best_hyperpars_vals_dict)), f)

    #Printing the saved hyper parameter values
    print(f'Saving the best hyperparameter values in tuning_stability.yaml for further optimization: \n{OmegaConf.to_yaml(dict(best_hyperpars_vals_dict))}\n')
    print('Please manually review if the hyperparameter values were selected correctly and adjust them in the tuning_stability.yaml file.')
    

def get_best_stability_paramset(stability_df, hyperparams_names):
    
    params_to_save = dict()
    
    stability_df_sorted = stability_df.sort_values('difference', ascending=False).iloc[:1]
    for hyperparam in hyperparams_names: 
        params_to_save[hyperparam] = stability_df_sorted[hyperparam].item()
        
    return(params_to_save, stability_df_sorted)

def get_best_4_latent_spaces(results_df_sorted):
    best_latent = []
    for index, row in results_df_sorted.iterrows():
        if row['num_latent'] not in best_latent:
            best_latent.append(int(row['num_latent']))
        if len(best_latent) >= 2:
            break
            
    #Adding two values from both sides 
    best_hypers_diff = max(best_latent) - min(best_latent)
    diff_from_zero = int(min(best_latent)/2)
    
    if min(best_latent) - best_hypers_diff > diff_from_zero:
        best_latent.append(min(best_latent) - best_hypers_diff)
        best_latent.append(max(best_latent) + best_hypers_diff)
    else:
        best_latent.append(min(best_latent) - diff_from_zero)
        best_latent.append(max(best_latent) + diff_from_zero)
    return(best_latent)

def make_and_save_best_stability_params(results_df, hyperparams_names, nepochs):
    
    print('Starting calculating the best hyperparameter values used in further model trainings') 
    
    # Getting best set of hyperparameters
    params_to_save, results_df_sorted = get_best_stability_paramset(results_df, hyperparams_names)
    params_to_save['tuned_num_epochs'] = nepochs

    # Saving best set of hyperparameters    
    with open('training_latent.yaml', "w") as f:
        OmegaConf.save(OmegaConf.create(dict(params_to_save)), f)
        
    # Printing the configuration saved 
    print(f'Saving best hyperparameter values in training_latent.yaml: \n {OmegaConf.to_yaml(dict(params_to_save))}')
    
    # Getting the latent spaces for training_association script and using them with the best hyperparam set
    best_latent = get_best_4_latent_spaces(results_df_sorted)
    params_to_save['num_latent'] = list(best_latent)
    
    # Saving best set of hyperparameters for training_association script
    with open('training_association.yaml', "w") as f:
        OmegaConf.save(OmegaConf.create(dict(params_to_save)), f)

    # Printing the configuration saved 
    print(f'Saving best hyperparameter values in training_association.yaml: \n{OmegaConf.to_yaml(dict(params_to_save))}')
