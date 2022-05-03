import numpy as np
from collections import defaultdict
import yaml

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

def read_yaml(file_name):
    with open(rf'{file_name}.yaml') as file:
        data_dict = yaml.load(file, Loader=yaml.FullLoader)
    return(data_dict)

def get_data(data_dict):
    
    # Define variables
    data_path = data_dict['path'] 
    categorical_data = data_dict['categorical_data_files']
    continuous_data = data_dict['continuous_data_files']
    data_of_interest = data_dict['data_of_interest']
    
    # Initiate lists
    cat_list, cat_names, con_list, con_names = [], [], [], []
    
    # Get categorical variables
    for cat_data in categorical_data:
        cat, cat_input = read_cat(data_path + f"{cat_data}.npy")
        cat_h = read_header(data_path + f"{cat_data}.tsv")
        
        cat_list.append(cat)
        cat_names.append(cat_h)
         
    # Get continuous variables
    for con_data in continuous_data:
        con, con_mask = read_con(data_path + f"{con_data}.npy")
        con_h = read_header(data_path + f"{con_data}.tsv", con_mask)
        
        con_list.append(con)
        con_names.append(con_h)
    
    #Change data types
    headers_all = tuple(cat_names+con_names)
    con_names = np.concatenate(con_names)
    cat_names = np.concatenate(cat_names)

    # Select dataset of interest
    if data_of_interest in categorical_data:
        drug, drug_input = read_cat(data_path + f"{data_of_interest}.npy")
        drug_h = read_header(data_path + f"{data_of_interest}.tsv")
    elif data_of_interest in continuous_data:
        drug, drug_mask = read_con(data_path + f"{data_of_interest}.npy")
        drug_h = read_header(data_path + f"{data_of_interest}.tsv", drug_mask)  
    else:
        raise ValueError("""In data.yaml file data_of_interest is chosen neither
                         from defined continuous nor categorical data types""")
    
    return(cat_list, con_list, cat_names, con_names, headers_all, drug, drug_h)
