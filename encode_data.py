#!/usr/bin/env python
import numpy as np
from collections import defaultdict

def encode_cat(f, num_classes=None, uniques=None):
   raw_input = list()
   ids = list()
   
   # read the file
   with open(f, "r") as f:
      header = f.readline()
      for line in f:
         line = line.rstrip()
         tmp = line.split("\t")
         raw_input.append(tmp[1:])
         ids.append(tmp[0])
   
   matrix = np.array(raw_input)
   n_labels = matrix.shape[1]
   n_samples = matrix.shape[0]
   
   # make endocding dict
   encodings = defaultdict(dict)
   count = 0
   
   if uniques is None:
      encodings = defaultdict(dict)
      for lab in range(0,n_labels):
         uniques = np.unique(matrix[:,lab])
         uniques = sorted(uniques)
         num_classes = len(uniques[uniques != "NA"])
         count = 0
         for u in uniques:
            if u == "NA":
               encodings[lab][u] = np.zeros(num_classes)
               continue
            encodings[lab][u] = np.zeros(num_classes)
            encodings[lab][u][count] = 1
            count += 1
   else:
      for u in uniques:
         if u == "NA":
            encodings[u] = np.zeros(num_classes)
            continue
         encodings[u] = np.zeros(num_classes)
         encodings[u][count] = 1
         count += 1
   
   
   # encode the data
   data_input = np.zeros((n_samples,n_labels,num_classes))
   i = 0
   for patient in matrix:
      
      data_sparse = np.zeros((n_labels, num_classes))
      
      count = 0
      for lab in patient:
         if uniques is None:
            data_sparse[count] = encodings[count][lab]
         else:
            data_sparse[count] = encodings[lab]
         count += 1
      
      data_input[i] = data_sparse
      i += 1
      
   return data_input

def encode_con(f):
   
   # read the file
   raw_input = list()
   with open(f, "r") as f:
      header = f.readline()
      for line in f:
         line = line.rstrip()
         tmp = np.array(line.split("\t"))
         vals = tmp[1:]
         vals[vals == 'NA'] = np.nan
         vals = list(map(float, vals))
         raw_input.append(vals)
   
   matrix = np.array(raw_input)
   n_labels = matrix.shape[1]
   n_samples = matrix.shape[0]
   headers = header.split("\t")[1:]
   
   mean = np.nanmean(matrix, axis=0)
   std = np.nanstd(matrix, axis=0)
    
   # z-score normalize
   data_input = matrix
   data_input -= mean
   data_input /= std
   
   return data_input
