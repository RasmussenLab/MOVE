# Import functions
import os
import torch
import numpy as np
import copy
import pandas as pd 

from torch.utils.data import DataLoader

import umap.umap_ as umap
import random
import itertools 

from move.models import vae
from move.utils import dataloaders
from move.utils.model_utils import get_latent, cal_recon, cal_cat_recon, cal_con_recon, get_baseline, change_drug, cal_sig_hits, correction_new, get_start_end_positions 
from move.utils.data_utils import initiate_default_dicts
from move.utils.seed import set_global_seed


def optimize_reconstruction(nHiddens, nLatents, nLayers, nDropout, nBeta, batch_sizes, nepochs, repeat, lrate, kldsteps, batchsteps, patience, cuda, path, cat_list, con_list, continuous_weights, categorical_weights, seed):
    """
    Performs hyperparameter tuning in terms of reconstruction
    
    inputs:
    returns:
    """ 

    
    # Preparing the data
    isExist = os.path.exists(path + 'hyperparameters/')
    if not isExist:
        os.makedirs(path + 'hyperparameters/')

    # Divide into test and training set
    npatient = cat_list[0].shape[0]
    train = random.sample(range(npatient), int(npatient*0.90))
    test = list(set(range(npatient)) - set(train))

    cat_list_train = []
    con_list_train = []
    cat_list_test = []
    con_list_test = []

    # Splitting into train and test sets
    for cat_data in cat_list:
        cat_list_train.append(cat_data[train])
        cat_list_test.append(cat_data[test])
    for con_data in con_list:
        con_list_train.append(con_data[train])
        con_list_test.append(con_data[test]) 

    # The selections are are saved
    np.save(path + "hyperparameters/train1.npy", np.array(train))
    np.save(path + "hyperparameters/test1.npy", np.array(test))

     # Create objects to save results
    latents, con_recons, cat_recons, recon_acc, loss_train,\
    likelihoods, best_epochs, latents_tests, con_recons_tests,\
    cat_recons_tests, recon_acc_tests, loss_tests, likelihood_tests = initiate_default_dicts(0, 13)

    # Getting the data
    mask_test, test_loader = dataloaders.make_dataloader(cat_list=cat_list_test, con_list=con_list_test, batchsize=1)
    test_loader = DataLoader(dataset=test_loader.dataset, batch_size=1, drop_last=False, shuffle=False) #, num_workers=1, pin_memory=test_loader.pin_memory

    results_df = []
    
    print('Beginning the hyperparameter tuning for reconstruction.\n')
    iters = itertools.product(nHiddens, nLatents, nLayers, nDropout, nBeta, batch_sizes, range(repeat))
    for nHidden, nLatent, nl, drop, b, batch_size, r in iters:
        combi = str([nHidden] * nl) + "+" + str(nLatent) + ", drop: " + str(drop) +", b: " + str(b) + ", batch: " + str(batch_size)

        best_model, loss, ce, sse, KLD, train_loader, mask, kld_w, cat_shapes, con_shapes, best_epoch = train_model(cat_list, con_list, categorical_weights, continuous_weights, batch_size, nHidden, nl, nLatent, b, drop, cuda, kldsteps, batchsteps, nepochs, lrate, seed, test_loader, patience, early_stopping=True)   


        # get results
        con_recon, train_test_loader, latent, latent_var, cat_recon, cat_class, loss, likelihood, latent_test, latent_var_test, cat_recon_test, cat_class_test, con_recon_test, loss_test, likelihood_test = get_latent(best_model, train_loader, test_loader, kld_w)

        # Calculate reconstruction
        cat_true_recon,true_recon,cat_true_recon_test,true_recon_test = cal_recon(cat_shapes, cat_recon, cat_class, train_loader, con_recon, con_shapes, cat_recon_test, cat_class_test, test_loader, con_recon_test)

        # Save output
        recon_acc[combi].append(cat_true_recon + true_recon)
        latents[combi].append(latent)
        con_recons[combi].append(con_recon)
        cat_recons[combi].append(cat_recon)
        loss_train[combi].append(loss)
        likelihoods[combi].append(likelihood)
        best_epochs[combi].append(best_epoch)

        recon_acc_tests[combi].append(cat_true_recon_test + true_recon_test)
        latents_tests[combi].append(latent_test)
        con_recons_tests[combi].append(con_recon_test)
        cat_recons_tests[combi].append(cat_recon_test)
        loss_tests[combi].append(loss_test)
        likelihood_tests[combi].append(likelihood_test)
        
        results_df.append({
            'num_hidden': nHidden,
            'num_latent': nLatent,
            'num_layers': nl,
            'dropout': drop,
            'beta': b,
            'batch_sizes': batch_size, 
            'repeats': r,
            'recon_acc': cat_true_recon + true_recon, 
            'latents': np.array(latent), 
            'con_recons': np.array(con_recon),
            'cat_recons': np.array(cat_recon),
            'loss_train': np.array(loss),
            'likelihoods': np.array(likelihood),
            'best_epochs': np.array(best_epoch),
            'recon_acc_test': np.array(cat_true_recon_test + true_recon_test),
            'latents_test': np.array(latent_test),
            'con_recons_test': np.array(con_recon_test),
            'cat_recons_test': np.array(cat_recon_test),
            'loss_test': np.array(loss_test),
            'likelihood_test': np.array(likelihood_test)
        })
        
    results_df = pd.DataFrame(results_df)
    
    print('\nFinished the hyperparameter tuning for reconstruction. Saving the results.')
    # Save output
    np.save(path + "hyperparameters/latent_benchmark_final.npy", latents)
    np.save(path + "hyperparameters/con_recon_benchmark_final.npy", con_recons)
    np.save(path + "hyperparameters/cat_recon_benchmark_final.npy", cat_recons)
    np.save(path + "hyperparameters/recon_acc_benchmark_final.npy", recon_acc)
    np.save(path + "hyperparameters/loss_benchmark_final.npy", loss_train)
    np.save(path + "hyperparameters/likelihood_benchmark_final.npy", likelihoods)
    np.save(path + "hyperparameters/best_epochs_benchmark_final.npy", best_epochs)

    np.save(path + "hyperparameters/test_latent_benchmark_final.npy", latents_tests)
    np.save(path + "hyperparameters/test_con_recon_benchmark_final.npy", con_recons_tests)
    np.save(path + "hyperparameters/test_cat_recon_benchmark_final.npy", cat_recons_tests)
    np.save(path + "hyperparameters/test_recon_acc_benchmark_final.npy", recon_acc_tests)
    np.save(path + "hyperparameters/test_loss_benchmark_final.npy", loss_tests)
    np.save(path + "hyperparameters/test_likelihood_benchmark_final.npy", likelihood_tests)
    
    print('The results saved.\n')
    
    return(likelihood_tests, recon_acc_tests, recon_acc, results_df)

    
    
def optimize_stability(nHiddens, nLatents, nDropout, nBeta, repeat, nepochs, nLayers, batch_sizes, lrate, kldsteps, batchsteps, cuda, path, con_list, cat_list, continuous_weights, categorical_weights, seed):
    
    models, latents, embeddings, con_recons, cat_recons, recon_acc, los, likelihood = initiate_default_dicts(1, 7)
    
    print('Beginning the hyperparameter tuning for stability.\n')
    iters = itertools.product(nHiddens, nLatents, nLayers, nDropout, nBeta, range(repeat))
    for nHidden, nLatent, nl, drop, b, r in iters:
        combi = str([nHidden] * nl) + "+" + str(nLatent) + ", do: " + str(drop) +", b: " + str(b)
        print(combi)

        best_model, loss, ce, sse, KLD, train_loader, mask, kld_w, cat_shapes, con_shapes, best_epoch = train_model(cat_list, con_list, categorical_weights, continuous_weights, batch_sizes, nHidden, nl, nLatent, b, drop, cuda, kldsteps, batchsteps, nepochs, lrate, seed, test_loader=None, patience=None, early_stopping=False)


        test_loader = DataLoader(dataset=train_loader.dataset, batch_size=1, drop_last=False,
                     shuffle=False, pin_memory=train_loader.pin_memory) #num_workers=1,

        latent, latent_var, cat_recon, cat_class, con_recon, test_loss, test_likelihood = best_model.latent(test_loader, kld_w)
        con_recon = np.array(con_recon)
        con_recon = torch.from_numpy(con_recon)

        cat_true_recon = cal_cat_recon(cat_shapes, cat_recon, cat_class)
        true_recon = cal_con_recon(train_loader, con_recon, con_shapes)

        ### Umap clustering 
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(latent)

        # save
        recon_acc[combi].append(cat_true_recon + true_recon)
        latents[combi].append(latent)
        embeddings[combi].append(embedding)
        con_recons[combi].append(con_recon)
        cat_recons[combi].append(cat_recon)
        

    # Saving the results
    print('\nFinished the hyperparameter tuning for stability. Saving the results.')
    
    np.save(path + "hyperparameters/embedding_stab.npy", embeddings)
    np.save(path + "hyperparameters/latent_stab.npy", latents)
    np.save(path + "hyperparameters/con_recon_stab.npy", con_recons)
    np.save(path + "hyperparameters/cat_recon_stab.npy", cat_recons)
    np.save(path + "hyperparameters/recon_acc_stab.npy", recon_acc)
    
    print('The results saved.\n')
    
    return(embeddings, latents, con_recons, cat_recons, recon_acc)


def train_model_association(path, cuda, nepochs, nLatents, batch_sizes, nHidden, nl, nBeta, drop, con_list, cat_list, continuous_weights, categorical_weights, version, repeats, kldsteps, batchsteps, lrate, drug, categorical_names, data_of_interest, seed):

    results, recon_results, recon_results_1, mean_bas = initiate_default_dicts(n_empty_dicts=0, n_list_dicts=4)

    start, end = get_start_end_positions(cat_list, categorical_names, data_of_interest)


    iters = itertools.product(nLatents, range(repeats))

    # Running the framework    
    for nLatent, repeat in iters: 
        best_model, loss, ce, sse, KLD, train_loader, mask, kld_w, cat_shapes, con_shapes, best_epoch = train_model(cat_list, con_list, categorical_weights, continuous_weights, batch_sizes, nHidden, nl, nLatent, nBeta, drop, cuda, kldsteps, batchsteps, nepochs, lrate, seed, test_loader=None, patience=None, early_stopping=False)


        train_test_loader = DataLoader(dataset=train_loader.dataset, batch_size=train_loader.batch_size, drop_last=False,
                              shuffle=False, pin_memory=train_loader.pin_memory) #num_workers=1,

        latent, latent_var, cat_recon, cat_class, con_recon, loss, likelihood = best_model.latent(train_test_loader, kld_w)

        con_recon = np.array(con_recon)
        con_recon = torch.from_numpy(con_recon)

        mean_baseline = get_baseline(best_model, train_test_loader, con_recon, repeat=1, kld_w=kld_w)
        recon_diff, groups = change_drug(best_model, train_test_loader, con_recon, drug, start, end, kld_w) #TODO: when the drug data starts 
        stat = cal_sig_hits(recon_diff, groups, drug, mean_baseline, train_loader.dataset.con_all)

        results[nLatent].append(stat)
        recon_diff_corr = dict()
        for r_diff in recon_diff.keys():
            recon_diff_corr[r_diff] = recon_diff[r_diff] - np.abs(mean_baseline[groups[r_diff]])

        mean_bas[nLatent].append(mean_baseline)
        recon_results[nLatent].append(recon_diff_corr)
        recon_results_1[nLatent].append(recon_diff)


    # Saving results
    isExist = os.path.exists(path + 'wp2.2/sig_overlap/')
    if not isExist:
        os.makedirs(path + 'wp2.2/sig_overlap/')

    isExist = os.path.exists(path + 'results/')
    if not isExist:
        os.makedirs(path + 'results/')

    with open(path + "results/results_" + version + ".npy", 'wb') as f:
        np.save(f, results)
    with open(path + "results/results_recon_" + version + ".npy", 'wb') as f:
        np.save(f, recon_results)
    with open(path + "results/results_groups_" + version + ".npy", 'wb') as f:
        np.save(f, groups)
    with open(path + "results/results_recon_mean_baseline_" + version + ".npy", 'wb') as f:
        np.save(f, mean_bas)
    with open(path + "results/results_recon_no_corr_" + version + ".npy", 'wb') as f:
        np.save(f, recon_results_1)

    cor_results = correction_new(results)

    with open(path + "wp2.2/sig_overlap/cor_results_" + version + ".npy", 'wb') as f:
        np.save(f, cor_results)
        
    
    
def train_model(cat_list, con_list, categorical_weights, continuous_weights, batch_size, nHidden, nl, nLatent, b, drop, cuda, kldsteps, batchsteps, nepochs, lrate, seed, test_loader, patience, early_stopping):
    
    device = torch.device("cuda" if cuda == True else "cpu")
    
    if seed is not None:
        set_global_seed(seed)
    
    # Initiate loader
    mask, train_loader = dataloaders.make_dataloader(cat_list=cat_list, 
                                                     con_list=con_list, 
                                                     batchsize=batch_size)


    #          ncategorical = train_loader.dataset.cat_all.shape[1]
    #          ncontinuous = train_loader.dataset.con_all.shape[1]
    con_shapes = train_loader.dataset.con_shapes
    cat_shapes = train_loader.dataset.cat_shapes

    model = vae.VAE(continuous_shapes=con_shapes, 
                 categorical_shapes=cat_shapes, 
                 num_hidden=[nHidden]*nl,
                 num_latent=nLatent,  
                 beta=b, 
                 continuous_weights=continuous_weights,
                 categorical_weights=categorical_weights, 
                 dropout=drop, 
                 cuda=cuda).to(device) # removed alpha=0.01,       

    # Create lists for saving loss
    loss = list();ce = list();sse = list();KLD = list();loss_test = list()


    l = len(kldsteps) # TODOs: not really clear this l
    rate = 20/l 
    kld_w = 0
    update = 1 + rate
    l_min = float('inf')
    count = 0

    # Train model
    for epoch in range(1, nepochs + 1):
        if epoch in kldsteps:
            kld_w = 1/20 * update
            #lrate = lrate - lrate*0.1
            update += rate

        if epoch in batchsteps:
            train_loader = DataLoader(dataset=train_loader.dataset,batch_size=int(train_loader.batch_size * 1.5),shuffle=True,drop_last=True) #,num_workers=train_loader.num_workers,pin_memory=train_loader.pin_memory  Is it really?: batch_size=int(train_loader.batch_size * 1.5)

        l, c, s, k = model.encoding(train_loader, epoch, lrate, kld_w)


        loss.append(l)
        ce.append(c)
        sse.append(s)
        KLD.append(k)

        if early_stopping:

            out = model.latent(test_loader, kld_w)
            loss_test.append(out[-2])
            print("Likelihood: " + str(out[-1]))


            if out[-1] > l_min and count < patience: # Change to infinite value
                count+=1
                
                if count % 5 == 0:
                    lrate = lrate - lrate*0.10


            elif count == patience: #Changed from 100
                break

            else:
                l_min = out[-1]
                best_model = copy.deepcopy(model)
                best_epoch = epoch
                count = 0

                if epoch > 3: # Added
                    break      # Added

            if epoch > 10: # Added
                break      # Added    
        else:
            best_model = copy.deepcopy(model)
            best_epoch = None
        #                loss_test = None 

    return(best_model, loss, ce, sse, KLD, train_loader, mask, kld_w, cat_shapes, con_shapes, best_epoch)        