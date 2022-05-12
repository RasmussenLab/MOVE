#!/usr/bin/env python

import torch
import numpy as np

from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

class Dataset(TensorDataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, cat_all=None, con_all=None, con_shapes=None, cat_shapes=None):
        'Initialization'
        #self.IDs = IDs
        if not (cat_all is None):
          self.cat_all = cat_all
          self.cat_shapes = cat_shapes
          self.npatients = cat_all.shape[0]
        else:
          self.cat = None
        if not (con_all is None):
          self.con_all = con_all
          self.npatients = con_all.shape[0]
          self.con_shapes = con_shapes
        else:
          self.con_all = None
        
  def __len__(self):
        'Denotes the total number of samples'
        return self.npatients

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        #ID = self.IDs[index]

        # Load data and get label
        if not (self.cat_all is None):
          cat_all_data = self.cat_all[index]
        else:
          cat_all_data = 0
        
        if not(self.con_all is None):
          con_all_data = self.con_all[index]
        else:
          con_all_data = 0
          
        return cat_all_data, con_all_data

def concat_cat_list(cat_list):
  n_cat = 0
  cat_shapes = list()
  first = 0
  
  for cat_d in cat_list:
    cat_shapes.append(cat_d.shape)
    cat_input = cat_d.reshape(cat_d.shape[0], -1)
    
    if first == 0:
      cat_all = cat_input
      del cat_input
      first = 1
    else:
      cat_all = np.concatenate((cat_all, cat_input), axis=1)
  
  # Make mask for patients with no measurments
  catsum = cat_all.sum(axis=1)
  mask = catsum > 5
  del catsum
  return cat_shapes, mask, cat_all

def concat_con_list(con_list, mask):
  n_con_shapes = []
  
  first = 0
  for con_d in con_list:
    
    n_con_shapes.append(con_d.shape[1])
      
    if first == 0:
      con_all = con_d
      first = 1
    else:
      con_all = np.concatenate((con_all, con_d), axis=1)
  
  consum = con_all.sum(axis=1)
  mask &= consum != 0
  del consum
  return n_con_shapes, mask, con_all

def make_dataloader(cat_list=None, con_list=None, batchsize=10, cuda=False):
    """Create a DataLoader for input of each data type - categorical,
    continuous and potentially each omcis set (currently proteomics, target
    metabolomicas, untarget metabolomics and transcriptomics). 

    Inputs:
        cat_list: list of categorical input matrix (N_patients x N_variables x N_max-classes)
        con_list: list of normalized continuous input matrix (N_patients x N_variables)
        batchsize: Starting size of minibatches for dataloader
        cuda: Pagelock memory of dataloader (use when using GPU acceleration)

    Outputs:
        DataLoader: An object feeding data to the VAE
    """
    
    if (cat_list is None and con_list is None):
      raise ValueError('At least one type of data must be in the input')
    
    # Handle categorical data sets
    if not (cat_list is None):
       cat_shapes, mask, cat_all= concat_cat_list(cat_list)

    else:
      mask = [True] * len(con_list[0])
    
    # Concetenate con datasetsand make final mask
    if not (con_list is None):
      n_con_shapes, mask, con_all = concat_con_list(con_list, mask)

    
    # Create dataset
    if not (cat_list is None or con_list is None):
      cat_all = cat_all[mask]
      con_all = con_all[mask]
      
      cat_all = torch.from_numpy(cat_all)
      con_all = torch.from_numpy(con_all)

      dataset = Dataset(con_all=con_all, con_shapes=n_con_shapes, cat_all=cat_all, cat_shapes=cat_shapes)
        
    elif not (con_list is None):
      con_all = con_all[mask]
      con_all = torch.from_numpy(con_all)
      dataset = Dataset(con_all=con_all, con_shapes=n_con_shapes)
    elif not (cat_list is None):
      cat_all = cat_all[mask]
      cat_all = torch.from_numpy(cat_all)
      dataset = Dataset(cat_all=cat_all, cat_shapes=cat_shapes)
    # Create dataloader
    dataloader = DataLoader(dataset=dataset, batch_size=batchsize, drop_last=False,
                             shuffle=True) #Changed num_workers and pin_memory; Changed drop_last to False
    return mask, dataloader
  
class VAE(nn.Module):
    """Variational autoencoder, subclass of torch.nn.Module.
    
    Instantiate with:
        ncategorical: Length of categorical variabel encoding if any
        ncontinuous: Number of continuous variables if any
        con_shapes: shape of the different continuous datasets if any
        cat_shapes: shape of the different categorical datasets if any
        nhiddens: List of n_neurons in the hidden layers [[200, 200]]
        nlatent: Number of neurons in the latent layer [15]
        beta: Multiply KLD by the inverse of this value [0.0001]
        con_weights: list of weights for each continuous dataset
        cat_weights: list of weights for each categorical dataset
        dropout: Probability of dropout on forward pass [0.2]
        cuda: Use CUDA (GPU accelerated training) [False]
    
    vae.trainmodel(dataloader, nepochs batchsteps, lrate, logfile, modelfile)
        Trains the model, returning None

    vae.encode(self, data_loader):
        Encodes the data in the data loader and returns the encoded matrix.
    """
    
    def __init__(self, ncategorical=None, ncontinuous=None, con_shapes=None, cat_shapes=None,
                 con_weights=None, cat_weights=None, nhiddens=[200,200], nlatent=20,
                 beta=0.01, dropout=0.2, cuda=False):
      
      if nlatent < 1:
        raise ValueError('Minimum 1 latent neuron, not {}'.format(nlatent))

      if beta <= 0:
        raise ValueError('beta must be > 0')
      
      if not (0 <= dropout < 1):
        raise ValueError('dropout must be 0 <= dropout < 1')
      
      if (ncategorical is None and ncontinuous is None):
        raise ValueError('At least one type of data must be in the input')
      
      if (con_shapes is None and cat_shapes is None):
        raise ValueError('Shapes of the input data must be provided')
      
      self.input_size = 0
      if not (ncontinuous is None or con_shapes is None):
        self.ncontinuous = ncontinuous
        self.input_size += self.ncontinuous
        self.con_shapes = con_shapes
        
        if not (con_weights is None):
          self.con_weights = con_weights
          if not len(con_shapes) == len(con_weights):
            raise ValueError('Number of continuous weights must be the same as number of continuous datasets')
      else:
        self.ncontinuous = None
      
      if not (ncategorical is None or cat_shapes is None):
        self.ncategorical = ncategorical
        self.input_size += self.ncategorical
        self.cat_shapes = cat_shapes
        
        if not (cat_weights is None):
          self.cat_weights = cat_weights
          if not len(cat_shapes) == len(cat_weights):
            raise ValueError('Number of categorical weights must be the same as number of categorical datasets')
      else:
        self.ncategorical = None
        
      super(VAE, self).__init__()
      
      # Initialize simple attributes
      self.usecuda = cuda
      self.beta = beta
      self.nhiddens = nhiddens
      self.nlatent = nlatent
      self.dropout = dropout
      
      self.device = torch.device("cuda" if self.usecuda == True else "cpu")
      
      # Activation functions
      self.relu = nn.LeakyReLU()
      self.softplus = nn.Softplus()
      self.sigmoid = nn.Sigmoid()
      self.log_softmax = nn.LogSoftmax(dim = 1)
      self.dropoutlayer = nn.Dropout(p=self.dropout)
      
      # Initialize lists for holding hidden layers
      self.encoderlayers = nn.ModuleList()
      self.encodernorms = nn.ModuleList()
      self.decoderlayers = nn.ModuleList()
      self.decodernorms = nn.ModuleList()
      
      ### Layers        
      # Hidden layers
      for nin, nout in zip([self.input_size] + self.nhiddens, self.nhiddens):
          self.encoderlayers.append(nn.Linear(nin, nout))
          self.encodernorms.append(nn.BatchNorm1d(nout))
      
      # Latent layers
      self.mu = nn.Linear(self.nhiddens[-1], self.nlatent) # mu layer
      self.var = nn.Linear(self.nhiddens[-1], self.nlatent) # logvariance layer
      
      # Decoding layers
      for nin, nout in zip([self.nlatent] + self.nhiddens[::-1], self.nhiddens[::-1]):
        self.decoderlayers.append(nn.Linear(nin, nout))
        self.decodernorms.append(nn.BatchNorm1d(nout))
      
      # Reconstruction - output layers
      self.out = nn.Linear(self.nhiddens[0], self.input_size) #to output
    
    def encode(self, tensor):
      tensors = list()

      # Hidden layers
      for encoderlayer, encodernorm in zip(self.encoderlayers, self.encodernorms):
        tensor = encodernorm(self.dropoutlayer(self.relu(encoderlayer(tensor))))
        tensors.append(tensor)
      
      #h1 = self.relu(self.fc1(x))
      return self.mu(tensor), self.var(tensor)

    def reparameterize(self, mu, logvar):
      std = torch.exp(0.5*logvar)
      eps = torch.randn_like(std)
      
      return eps.mul(std).add_(mu)
    
    def decompose_categorical(self, reconstruction):
      cat_tmp = reconstruction.narrow(1, 0, self.ncategorical)
      
      # handle soft max for each categorical dataset
      cat_out = []
      pos = 0
      for cat_shape in self.cat_shapes:
        cat_dataset = cat_tmp[:, pos:(cat_shape[1]*cat_shape[2] + pos)]
        
        cat_out_tmp = cat_dataset.view(cat_dataset.shape[0], cat_shape[1], cat_shape[2])
        cat_out_tmp = cat_out_tmp.transpose(1, 2)
        cat_out_tmp = self.log_softmax(cat_out_tmp)
        
        cat_out.append(cat_out_tmp)
        pos += cat_shape[1]*cat_shape[2]
        
      return cat_out
    
    def decode(self, tensor):
      tensors = list()

      for decoderlayer, decodernorm in zip(self.decoderlayers, self.decodernorms):
        tensor = decodernorm(self.dropoutlayer(self.relu(decoderlayer(tensor))))
        tensors.append(tensor)    
      reconstruction = self.out(tensor)
      # Decompose reconstruction to categorical and continuous variables
      # if both types are in the input
      if not (self.ncategorical is None or self.ncontinuous is None):
        cat_out = self.decompose_categorical(reconstruction)
        con_out = reconstruction.narrow(1, self.ncategorical, self.ncontinuous)
      elif not (self.ncategorical is None):
        cat_out = self.decompose_categorical(reconstruction)
        con_out = None
      elif not (self.ncontinuous is None):
        cat_out = None
        con_out = reconstruction.narrow(1, 0, self.ncontinuous)
      
      return cat_out, con_out

    def forward(self, tensor):
      mu, logvar = self.encode(tensor)
      z = self.reparameterize(mu, logvar)
      cat_out, con_out = self.decode(z)

      return cat_out, con_out, mu, logvar
    
    def calculate_cat_error(self, cat_in, cat_out):
      batch_size = cat_in.shape[0]
      
      # calcualte target values for all cat datasets
      count = 0
      cat_errors = []
      pos = 0
      for cat_shape in self.cat_shapes:
        cat_dataset = cat_in[:, pos:(cat_shape[1]*cat_shape[2] + pos)]
        
        cat_dataset = cat_dataset.view(cat_in.shape[0], cat_shape[1], cat_shape[2])
        cat_target = cat_dataset
        cat_target = cat_target.argmax(2)
        cat_target[cat_dataset.sum(dim = 2) == 0] = -1
        cat_target = cat_target.to(self.device)
        
        # Cross entropy loss for categroical
        loss = nn.NLLLoss(reduction='sum',  ignore_index = -1)
        cat_errors.append(loss(cat_out[count], cat_target) / (batch_size * cat_shape[1]))
        count += 1
        pos += cat_shape[1]*cat_shape[2]
      
      cat_errors = torch.stack(cat_errors)
      return cat_errors
    
    def calculate_con_error(self, con_in, con_out, loss):
      batch_size = con_in.shape[0]
      total_shape = 0
      con_errors = []
      for s in self.con_shapes:
        c_in = con_in[:,total_shape:(s + total_shape - 1)]
        c_re = con_out[:,total_shape:(s + total_shape - 1)]
        error = loss(c_re, c_in) / batch_size
        con_errors.append(error)
        total_shape += s
      
      con_errors = torch.stack(con_errors)
      con_errors = con_errors / torch.Tensor(self.con_shapes)
      MSE = torch.sum(con_errors * torch.Tensor(self.con_weights))
      return MSE
    
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, cat_in, cat_out, con_in, con_out, mu, logvar, kld_w):
      MSE = 0
      CE = 0
      # calculate loss for catecorical data if in the input
      if not (cat_out is None):
        cat_errors = self.calculate_cat_error(cat_in, cat_out)
        if not (self.cat_weights is None):
          CE = torch.sum(cat_errors * torch.Tensor(self.cat_weights))
        else:
          CE = torch.sum(cat_errors) / len(cat_errors)
      
      # calculate loss for continuous data if in the input
      if not (con_out is None):
        batch_size = con_in.shape[0]
        # Mean square error loss for continauous
        loss = nn.MSELoss(reduction='sum')
        # set missing data to 0 to remove any loss these would provide
        con_out[con_in == 0] == 0
        
        # include different weights for each omics dataset
        if not (self.con_weights is None):
          MSE = self.calculate_con_error(con_in, con_out, loss)
        else:
          MSE = loss(con_out, con_in) / (batch_size *self.ncontinuous)
      
      # see Appendix B from VAE paper:
      # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
      # https://arxiv.org/abs/1312.6114
      # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
      KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (batch_size)
      
      KLD_weight =  self.beta * kld_w
      loss = CE + MSE + KLD * KLD_weight
      
      return loss, CE, MSE, KLD * KLD_weight
    
    def encoding(self, train_loader, epoch, lrate, kld_w):
        self.train()
        train_loss = 0
        log_interval = 50
        
        optimizer = optim.Adam(self.parameters(), lr=lrate)

        epoch_loss = 0
        epoch_kldloss = 0
        epoch_sseloss = 0
        epoch_bceloss = 0
        
        for batch_idx, (cat, con) in enumerate(train_loader):
            # Move input to GPU if requested
            cat = cat.to(self.device)
            con = con.to(self.device)
              
            if not (self.ncategorical is None or self.ncontinuous is None):
              tensor = torch.cat((cat, con), 1)
            elif not (self.ncategorical is None):
              tensor = cat
            elif not (self.ncontinuous is None):
              tensor = con
            
            optimizer.zero_grad()
            
            cat_out, con_out, mu, logvar = self(tensor)
 
            loss, bce, sse, kld = self.loss_function(cat, cat_out, con, con_out, mu, logvar, kld_w)
            loss.backward()
            
            epoch_loss += loss.data.item()
            epoch_kldloss += kld.data.item()
            
            if not (self.ncontinuous is None):
              epoch_sseloss += sse.data.item()
            
            if not (self.ncategorical is None):
              epoch_bceloss += bce.data.item()
            
            optimizer.step()
            
        print('\tEpoch: {}\tLoss: {:.6f}\tCE: {:.7f}\tSSE: {:.6f}\tKLD: {:.4f}\tBatchsize: {}'.format(
                  epoch ,
                  epoch_loss / len(train_loader),
                  epoch_bceloss / len(train_loader),
                  epoch_sseloss / len(train_loader),
                  epoch_kldloss / len(train_loader),
                  train_loader.batch_size,
                  ))
        return epoch_loss / len(train_loader), epoch_bceloss / len(train_loader), epoch_sseloss / len(train_loader), epoch_kldloss / len(train_loader)
    
    def make_cat_recon_out(self, length):
      cat_total_shape = 0
      for cat_shape in self.cat_shapes:
        cat_total_shape += cat_shape[1]
      
      cat_class = np.empty((length, cat_total_shape), dtype=np.int32)
      cat_recon = np.empty((length, cat_total_shape), dtype=np.int32)
      return cat_class, cat_recon, cat_total_shape
    
    def get_cat_recon(self, batch, cat_total_shape, cat, cat_out):
      count = 0
      cat_out_class = np.empty((batch, cat_total_shape), dtype=np.int32)
      cat_target = np.empty((batch, cat_total_shape), dtype=np.int32)
      pos = 0
      shape_1 = 0
      for cat_shape in self.cat_shapes:
        # Get input categorical data
        cat_in_tmp = cat[:, pos:(cat_shape[1]*cat_shape[2] + pos)]
        cat_in_tmp = cat_in_tmp.view(cat.shape[0], cat_shape[1], cat_shape[2])

        # Calculate target values for input
        cat_target_tmp = cat_in_tmp
        cat_target_tmp = np.argmax(cat_target_tmp.detach(), 2)
        cat_target_tmp[cat_in_tmp.sum(dim = 2) == 0] = -1
        cat_target[:,shape_1:(cat_shape[1] + shape_1)] = cat_target_tmp.numpy()
        
        # Get reconstructed categorical data
        cat_out_tmp = cat_out[count]
        cat_out_tmp = cat_out_tmp.transpose(1, 2)
        cat_out_class[:,shape_1:(cat_shape[1] + shape_1)] = np.argmax(cat_out_tmp, 2).numpy()
        
        # make counts for next dataset
        pos += cat_shape[1]*cat_shape[2]
        shape_1 += cat_shape[1]
        count += 1
      
      return cat_out_class, cat_target
    
    def latent(self, test_loader, kld_w):
        self.eval()
        test_loss = 0
        test_likelihood = 0
        
        length = test_loader.dataset.npatients
        latent = np.empty((length, self.nlatent), dtype=np.float32)
        latent_var = np.empty((length, self.nlatent), dtype=np.float32)
        
        # reconstructed output
        if not (self.ncategorical is None):
          cat_class, cat_recon, cat_total_shape = self.make_cat_recon_out(length)
        else:
          cat_class = None
          cat_recon = None
        
        if not (self.ncontinuous is None):
          con_recon = np.empty((length, self.ncontinuous), dtype=np.float32)
        else:
          con_recon = None
        
        row = 0
        with torch.no_grad():
            for (cat,con) in test_loader:
              cat = cat.to(self.device)
              con = con.to(self.device)
              cat.requires_grad = False
              con.requires_grad = False
              
              # get dataset
              if not (self.ncategorical is None or self.ncontinuous is None):
                tensor = torch.cat((cat, con), 1)
              elif not (self.ncategorical is None):
                tensor = cat
              elif not (self.ncontinuous is None):
                tensor = con
                
              # Evaluate
              cat_out, con_out, mu, logvar = self(tensor)
              con_out
              mu = mu.to(self.device)
              logvar = logvar.to(self.device)
              batch = len(mu)
              
              loss, bce, sse, kld = self.loss_function(cat, cat_out, con, con_out, mu, logvar, kld_w)
              test_likelihood += bce + sse
              test_loss += loss.data.item()
              
              if not (self.ncategorical is None):
                cat_out_class, cat_target = self.get_cat_recon(batch, cat_total_shape, cat, cat_out)
                cat_recon[row: row + len(cat_out_class)] = cat_out_class
                cat_class[row: row + len(cat_target)] =  cat_target
              
              if not (self.ncontinuous is None):
                con_recon[row: row + len(con_out)] = con_out
              
              latent_var[row: row + len(logvar)] = logvar
              latent[row: row + len(mu)] = mu
              row += len(mu)
        

           
        
        test_loss /= len(test_loader)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        assert row == length
        return latent, latent_var, cat_recon, cat_class, con_recon, test_loss, test_likelihood
