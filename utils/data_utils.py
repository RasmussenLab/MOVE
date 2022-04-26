import numpy as np
from collections import defaultdict

# Functions for loading data
def read_cat(file):
   data = np.load(file)
   data = data.astype(np.float32)
   data_input = data.reshape(data.shape[0], -1)
   
   return data, data_input

def read_con(file):
   data = np.load(file)
   data = data.astype(np.float32)
   data[np.isnan(data)] = 0
   consum = data.sum(axis=0)
   mask_col = consum != 0
   data = data[:,mask_col]
   
   return data, mask_col

def read_header(file, mask=None, start=1):
   with open(file, "r") as f:
      h = f.readline().rstrip().split("\t")[start:]
   
   if not mask is None:
      h = np.array(h)
      h = h[mask]
   
   return h

def initiate_default_dicts(n_empty_dicts, n_list_dicts):
   
   dicts = [defaultdict() for _ in range(n_empty_dicts)] + \
           [defaultdict(list) for _ in range(n_list_dicts)]
    
   return(tuple(dicts))