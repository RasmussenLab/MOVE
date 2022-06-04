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

import umap.umap_ as umap
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
from utils.model_utils import *
from utils.data_utils import *

def optimize_reconstruction(nHiddens, nLatents, nLayers, nDropout, nBeta, batch_sizes, nepochs, repeat, lrate, kldsteps, batchsteps, patience, cuda, path, cat_list, con_list):
    
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
   mask_test, test_loader = VAE_v2_1.make_dataloader(cat_list=cat_list_test, con_list=con_list_test, batchsize=1)
   test_loader = DataLoader(dataset=test_loader.dataset, batch_size=1, drop_last=False, shuffle=False) #, num_workers=1, pin_memory=test_loader.pin_memory

   mask, train_loader = VAE_v2_1.make_dataloader(cat_list=cat_list_train, con_list=con_list_train, batchsize=10)

   ncontinuous = train_loader.dataset.con_all.shape[1]
   con_shapes = train_loader.dataset.con_shapes

   ncategorical = train_loader.dataset.cat_all.shape[1]
   cat_shapes = train_loader.dataset.cat_shapes

   device = torch.device("cuda" if cuda == True else "cpu")
   l = len(kldsteps) # TODOs: not really clear this l
   rate = 20/l 
   
   iters = itertools.product(nHiddens, nLatents, nLayers, nDropout, nBeta, batch_sizes)
   for nHidden, nLatent, nl, drop, b, batch_size in iters:
      for r in range(repeat):
         combi = str([nHidden] * nl) + "+" + str(nLatent) + ", drop: " + str(drop) +", b: " + str(b) + ", batch: " + str(batch_size)
         
         # Initiate loader
         mask, train_loader = VAE_v2_1.make_dataloader(cat_list=cat_list_train, 
                                                       con_list=con_list_train, 
                                                       batchsize=batch_size)

         # Initate model
         model = VAE_v2_1.VAE(ncategorical=ncategorical, ncontinuous=ncontinuous,
                           con_shapes=con_shapes, cat_shapes=cat_shapes, nhiddens=[nHidden]*nl,
                           nlatent=nLatent,  beta=b, con_weights=[1,1,1,1,1,1,1],
                           cat_weights=[1,1,1], dropout=drop, cuda=cuda).to(device) # removed alpha=0.01,

         # Create lists for saving loss
         loss = list();ce = list();sse = list();KLD = list();loss_test = list()

         # Train model
         kld_w = 0
         update = 1 + rate
         l_min = None
         count = 0

        
         for epoch in range(1, nepochs + 1):
            if epoch in kldsteps:
               kld_w = 1/20 * update
               #lrate = lrate - lrate*0.1
               update += rate

            if epoch in batchsteps:
               train_loader = DataLoader(dataset=train_loader.dataset,batch_size=int(train_loader.batch_size * 1.5),shuffle=True,drop_last=True) #,num_workers=train_loader.num_workers,pin_memory=train_loader.pin_memory  Is it really?: batch_size=int(train_loader.batch_size * 1.5)

            l, c, s, k = model.encoding(train_loader, epoch, lrate, kld_w)

            out = model.latent(test_loader, kld_w)
            loss_test.append(out[-2])
            
            print("Likelihood: " + str(out[-1]))

            loss.append(l)
            ce.append(c)
            sse.append(s)
            KLD.append(k)

            if (l_min != None and out[-1] > l_min) and count < patience: # Change to infinite value
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

         # get results
         con_recon, train_test_loader, latent, latent_var, cat_recon, cat_class, loss, likelihood, latent_test, latent_var_test, cat_recon_test, cat_class_test, con_recon_test, loss_test, likelihood_test = get_latent(best_model, train_loader, test_loader, kld_w)

         # Calculate reconstruction\
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
            
        #          recon_acc, latents, con_recons, cat_recons, loss_train, likelihoods, best_epochs, recon_acc_tests,\
#          latents_tests, con_recons_tests, cat_recons_tests, loss_tests, likelihood_tests = save_input(\
#                 combi,cat_true_recon,true_recon,latent,con_recon,cat_recon,loss,likelihood,best_epoch,\
#                 cat_true_recon_test,true_recon_test,latent_test,con_recon_test,cat_recon_test,loss_test,likelihood_test)
        
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
    
# def get_list_value(*args):
#     arg_tuple = [arg[0] if len(arg) == 1 else arg for arg in args]
#     return(arg_tuple)
    
    
def optimize_stability(nHiddens, nLatents, drop_outs, repeat, nepochs, nl, batch_sizes, lrate, kldsteps, batchsteps, cuda, path, con_list, cat_list):
    
   device = torch.device("cuda" if cuda == True else "cpu") #TODO: repeat is not included anywhere, I think
   models, latents, embeddings, con_recons, cat_recons, recon_acc, los, likelihood = initiate_default_dicts(1, 7)
   
 
   iters = itertools.product(nHiddens, nLatents, drop_outs, range(repeat))
   for nHidden, nLatent, do, r in iters:
      combi = str([nHidden] * nl) + "+" + str(nLatent) + ", Drop-out:" + str(do)
      print(combi)

      mask, train_loader = VAE_v2_1.make_dataloader(cat_list=cat_list, con_list=con_list, batchsize=batch_sizes)

      ncategorical = train_loader.dataset.cat_all.shape[1]
      ncontinuous = train_loader.dataset.con_all.shape[1]
      con_shapes = train_loader.dataset.con_shapes
      cat_shapes = train_loader.dataset.cat_shapes

      model = VAE_v2_1.VAE(ncategorical=ncategorical, ncontinuous= ncontinuous,
                        con_shapes=con_shapes, cat_shapes=cat_shapes, nhiddens=[nHidden]*nl,
                        nlatent=nLatent,  beta=0.0001, con_weights=[1,1,1,1,1,1,1],
                        cat_weights=[1,1,1], dropout=do, cuda=cuda).to(device) #alpha=0.1, 

      loss = list(); ce = list(); sse = list(); KLD = list()
      
      kld_w = 0
      l = len(kldsteps)
      rate = 20/l 
      update = 1
    
      for epoch in range(1, nepochs + 1):

         if epoch in kldsteps:
            kld_w = 1/20 * update
            update += rate

         if epoch in batchsteps:
                  train_loader = DataLoader(dataset=train_loader.dataset,
                                          batch_size=int(train_loader.batch_size * 1.25), 
                                          shuffle=True,
                                          drop_last=True,
                                          num_workers=train_loader.num_workers,
                                          pin_memory=train_loader.pin_memory)
                
         l, c, s, k = model.encoding(train_loader, epoch, lrate, kld_w)
        
         loss.append(l)
         ce.append(c)
         sse.append(s)
         KLD.append(k)

      test_loader = DataLoader(dataset=train_loader.dataset, batch_size=1, drop_last=False,
                     shuffle=False, pin_memory=train_loader.pin_memory) #num_workers=1,

      latent, latent_var, cat_recon, cat_class, con_recon, test_loss, test_likelihood = model.latent(test_loader, kld_w)
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
   np.save(path + "hyperparameters/embedding_stab.npy", embeddings)
   np.save(path + "hyperparameters/latent_stab.npy", latents)
   np.save(path + "hyperparameters/con_recon_stab.npy", con_recons)
   np.save(path + "hyperparameters/cat_recon_stab.npy", cat_recons)
   np.save(path + "hyperparameters/recon_acc_stab.npy", recon_acc)
   
   return(embeddings, latents, con_recons, cat_recons, recon_acc)



def train_model_latent(path, cuda, nepochs, kldsteps, batchsteps, lrate, con_list, cat_list, nHiddens, nLatent, nl, beta, dropout):
    
    device = torch.device("cuda" if cuda == True else "cpu")
    
    # Making the dataloader
    mask, train_loader = VAE_v2_1.make_dataloader(cat_list=cat_list, con_list=con_list, batchsize=10) #Added drop_last

    # Get variabels needed to initialize the model
    ncontinuous = train_loader.dataset.con_all.shape[1]
    con_shapes = train_loader.dataset.con_shapes

    ncategorical = train_loader.dataset.cat_all.shape[1]
    cat_shapes = train_loader.dataset.cat_shapes
    
    # Make model
    model = VAE_v2_1.VAE(ncategorical=ncategorical, ncontinuous=ncontinuous, 
                         con_shapes=con_shapes, cat_shapes=cat_shapes,
                         nhiddens=[nHiddens]*nl, nlatent=nLatent, beta=beta, 
                         cat_weights=[1,1,1], con_weights=[2,1,1,1,1,1,1], 
                         dropout=dropout, cuda=cuda).to(device)
    
    kld_w = 0
    l = len(kldsteps)
    rate = 20/l
    update = 1 + rate
    
    # Lists for saving the results
    losses = list(); ce = list(); sse = list(); KLD = list()

    # Training the model
    for epoch in range(1, nepochs + 1):

        if epoch in kldsteps:
            kld_w = 1/20 * update
            update += r

        if epoch in batchsteps:
                train_loader = DataLoader(dataset=train_loader.dataset,
                                          batch_size=int(train_loader.batch_size * 1.25),
                                          shuffle=True,
                                          drop_last=False, #Added
                                          num_workers=train_loader.num_workers,
                                          pin_memory=train_loader.pin_memory)

        l, c, s, k = model.encoding(train_loader, epoch, lrate, kld_w)

        losses.append(l)
        ce.append(c)
        sse.append(s)
        KLD.append(k)

        best_model = copy.deepcopy(model)
    return(best_model, mask, train_loader, losses, ce, sse, KLD, cat_shapes, con_shapes)

def train_model_association(path, cuda, nepochs, nLatents, con_list, cat_list, version, repeats, kldsteps, batchsteps, lrate, drug):

   results, recon_results, recon_results_1, mean_bas = initiate_default_dicts(n_empty_dicts=0, n_list_dicts=4)

   device = torch.device("cuda" if cuda == True else "cpu")
   
   data_dict = read_yaml('data')
   start, end = get_start_end_positions(cat_list, data_dict)
    
   l = len(kldsteps)
   rate = 20/l
    
   iters = itertools.product(nLatents, range(repeats))
   # Running the framework
   for l, repeat in iters: 
      mask, train_loader = VAE_v2_1.make_dataloader(cat_list=cat_list, con_list=con_list, batchsize=10)

      ncontinuous = train_loader.dataset.con_all.shape[1]
      con_shapes = train_loader.dataset.con_shapes

      ncategorical = train_loader.dataset.cat_all.shape[1]
      cat_shapes = train_loader.dataset.cat_shapes

      model = VAE_v2_1.VAE(ncategorical=ncategorical, ncontinuous=ncontinuous, con_shapes=con_shapes,
                     cat_shapes=cat_shapes, nhiddens=[2000], nlatent=l,
                     beta=0.0001, cat_weights=[1,1,1], con_weights=[2,1,1,1,1,1,1], dropout=0.1, cuda=cuda).to(device)

      update = 1 + rate
      kld_w = 0
    
      for epoch in range(1, nepochs + 1):

         if epoch in kldsteps:
            kld_w = 1/20 * update
            update += rate

         if epoch in batchsteps:
                  train_loader = DataLoader(dataset=train_loader.dataset,
                                          batch_size=int(train_loader.batch_size * 1.5),
                                          shuffle=True,
                                          drop_last=True,
                                          pin_memory=train_loader.pin_memory) # num_workers=train_loader.num_workers,

         tmp_res = model.encoding(train_loader, epoch, lrate, kld_w)

      train_test_loader = DataLoader(dataset=train_loader.dataset, batch_size=train_loader.batch_size, drop_last=False,
                              shuffle=False, pin_memory=train_loader.pin_memory) #num_workers=1,

      latent, latent_var, cat_recon, cat_class, con_recon, loss, likelihood = model.latent(train_test_loader, kld_w)

      con_recon = np.array(con_recon)
      con_recon = torch.from_numpy(con_recon)

      mean_baseline = get_baseline(model, train_test_loader, con_recon, repeat=1, kld_w=kld_w)
      recon_diff, groups = change_drug(model, train_test_loader, con_recon, drug, start, end, kld_w) #TODO: when the drug data starts 
      stat = cal_sig_hits(recon_diff, groups, drug, mean_baseline, train_loader.dataset.con_all)

      results[l].append(stat)
      recon_diff_corr = dict()
      for r_diff in recon_diff.keys():
         recon_diff_corr[r_diff] = recon_diff[r_diff] - np.abs(mean_baseline[groups[r_diff]])

      mean_bas[l].append(mean_baseline)
      recon_results[l].append(recon_diff_corr)
      recon_results_1[l].append(recon_diff)
    
    
    
    # Saving results
   isExist = os.path.exists(path + 'wp2.2/sig_overlap/')
   if not isExist:
      os.makedirs(path + 'wp2.2/sig_overlap/')

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