import os 
import numpy as np
import pandas as pd
from collections import defaultdict
from omegaconf import OmegaConf

def merge_configs(base_config, config_types):
    """
    Merges base_config with user defined configuration
    
    inputs:
        base_config: YAML configuration
        config_types: list of ints of names of user defined configuration types 
    returns:
        config_section: YAML configuration of base_config overrided by user defined configs and filtered for config_types classes
    """
    user_config_dict = dict()
    override_types = []
    
    # Getting the user defined configs 
    for config_type in config_types:
        exist = os.path.isfile(config_type + '.yaml')
        if exist:    
            override_types.append(config_type + '.yaml')
            # Getting name of user config file and loading it 
            user_config_name = base_config[config_type]['user_config']
            user_config = OmegaConf.load(user_config_name)

            # Making dict with the same key as in configuration file
            user_config_dict[config_type] = user_config
    
    # Printing what was overrided
    override_types_str = ', '.join(str(override_type) for override_type in override_types)
    print(f'Overriding the default config with configs from {override_types_str}')
    
    # Merging the base and user defined config file
    config = OmegaConf.merge(base_config, user_config_dict)
    
    # Getting a subsection of data used for printing 
    config_section_dict = {x: config[x] for x in config_types if x in config}
    config_section = OmegaConf.create(config_section_dict)

    print(f'\nConfiguration used: \n---\n{OmegaConf.to_yaml(config_section)}---\n')
    return(config_section)


# Functions for loading data
def read_cat(path, file_name):
    """
    Reads categorical data file into numpy array
    
    inputs:
        path: pathway to the directory where file is located
        file_name: str of file name to read (in .npy format) 
    returns:
        data: np.array of input data
    """
    
    data = np.load(path + file_name)
    data = data.astype(np.float32)

    return data

def read_con(path, file_name):
    """
    Reads continuous data file into np.array, sets nan values as zeros and filters columns if all of the values were nan 
    
    inputs:
        path: pathway to the directory where file is located
        file_name: str of file name to read (in .npy format) 
    returns:
        data: np.array of input data
        mask_col: np.array of boolean objects, where False value corresponds to features that were filtered out
    """
    data = np.load(path + file_name)
    data = data.astype(np.float32)
    data[np.isnan(data)] = 0
    consum = np.absolute(data).sum(axis=0)
    mask_col = consum != 0
    data = data[:,mask_col]
    return data, mask_col

def read_header(path, file_name, mask=None, start=1):
    """
    Reads features names from the headers 
    inputs:
        path: pathway to the directory where file is located
        file_name: str of file name to read (in .npy format) 
        mask: np.array of boolean objects, where False value corresponds to features to filtered out
        start: int corresponding to how many lines to read for the header
    returns:
        header: np.array of strings of feature names
    """    
    with open(path + file_name, "r") as f:
        header = f.readline().rstrip().split("\t")[start:]
    if not mask is None:
        header = np.array(header)
        header = header[mask]
    
    return header

def initiate_default_dicts(n_empty_dicts, n_list_dicts):
    """
    Initiates empty default dictionaries
    
    inputs:
        n_empty_dicts: int of how many defaultdicts without specified data type to initiate
        n_list_dicts: int of how many defaultdicts with list type to initiate
    returns:
        tuple(default_dicts): tuple with initiated defaultdicts
    """   
    default_dicts = [defaultdict() for _ in range(n_empty_dicts)] + \
            [defaultdict(list) for _ in range(n_list_dicts)]
     
    return(tuple(default_dicts))

def get_data(raw_data_path, interim_data_path, categorical_data_names, continuous_data_names, data_of_interest):
    """
    Reads the data for models' inputs
    
    inputs:
        raw_data_path: str of pathway to raw data folder (e.g. .tsv)
        interim_data_path: str of a pathway to a folder of intermediate data files (e.g. .npy)
        categorical_data_names: list of strings of categorical data type names
        continuous_data_names: list of strings of continuous data type names
        data_of_interest: str of data type name whose features are changed to test their effects in the pipeline
    returns:
        cat_list: list of np.arrays for data of categorical data type 
        con_list: list of np.arrays for data of continuous data type
        cat_names: np.array of strings of feature names of continuous data 
        con_names: np.array of strings of feature names of categorical data 
        headers_all: np.array of strings of feature names of categorical data
        drug: np.array of input data whose feature data are changed to test their effects in the pipeline
        drug_h: np.array of strings of feature names data type whose data are changed to test their effects in the pipeline
    """   
        
    # Initiate lists
    cat_list, cat_names, con_list, con_names = [], [], [], []
     
    # Get categorical variables
    for cat_data in categorical_data_names:
        cat = read_cat(interim_data_path, f"{cat_data}.npy")
        cat_h = read_header(raw_data_path,  f"{cat_data}.tsv")
          
        cat_list.append(cat)
        cat_names.append(cat_h)

     # Get continuous variables
    for con_data in continuous_data_names:
        con, con_mask = read_con(interim_data_path, f"{con_data}.npy")
        con_h = read_header(raw_data_path, f"{con_data}.tsv", con_mask)
          
        con_list.append(con)
        con_names.append(con_h)
 
    #Change data types
    headers_all = tuple(cat_names+con_names)
    con_names = np.concatenate(con_names)
    cat_names = np.concatenate(cat_names)

    # Select dataset of interest
    if data_of_interest in categorical_data_names:
        drug = read_cat(interim_data_path, f"{data_of_interest}.npy")
        drug_h = read_header(raw_data_path, f"{data_of_interest}.tsv")
    elif data_of_interest in continuous_data_names:
        drug, drug_mask = read_con(interim_data_path, f"{data_of_interest}.npy")
        drug_h = read_header(raw_data_path, f"{data_of_interest}.tsv", drug_mask)  
    else:
        raise ValueError("""In data.yaml file data_of_interest is chosen neither
                                 from defined continuous nor categorical data types""")
     
    return(cat_list, con_list, cat_names, con_names, headers_all, drug, drug_h)


# Functions for encoding

def encode_cat(raw_input, na='NA'):
    """
    Encodes categorical data into one-hot encoding
    
    inputs:
         raw_input: a list of source data sorted by IDs from baseline_ids.txt file
    returns:
         data_input: one hot encoded data
    """ 
     
    matrix = np.array(raw_input)
    n_labels = matrix.shape[1]
    n_samples = matrix.shape[0]
    
    unique_sorted_data = np.unique(raw_input)
    num_classes = len(unique_sorted_data[~np.isnan(unique_sorted_data)])
    uniques = [*range(num_classes), 'nan']
 
    # make endocding dict
    encodings = defaultdict(dict)
    count = 0
    no_unique = 0
    
    for u in uniques:
        if u == na:
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
            if no_unique == 1:
                data_sparse[count] = encodings[count][lab]
            else:
                if lab != na:
                    lab = int(float(lab))
                data_sparse[count] = encodings[lab]
            count += 1
        
        data_input[i] = data_sparse
        i += 1
        
    return data_input

def encode_con(raw_input):
    """
    Log transforms and z-normalizes the data
    
    Input: 
         raw_input: a list of source data sorted by IDs from baseline_ids.txt file
    Returns:
         data_input: numpy array with log transformed and z-score normalized data
         mask_col: a np.array vector of Bolean values that correspond to nonzero sd values 
    """

    matrix = np.array(raw_input)
    consum = matrix.sum(axis=1)
    
    data_input = np.log2(matrix + 1) 
    
    # remove 0 variance
    std = np.nanstd(data_input, axis=0)
    mask_col = std != 0
    data_input = data_input[:,mask_col]
     
    # z-score normalize
    mean = np.nanmean(data_input, axis=0)
    std = np.nanstd(data_input, axis=0)
    data_input -= mean
    data_input /= std
    return data_input, mask_col 


def sort_data(data, ids, labels):
    """
    Sorts data based on the ids file
    
    Inputs:
         data: a dictionary with the data to encode
         ids: a list of personal identfiers (ID) from baseline_ids.txt file
         labels: a list of column names from the source data file
    Returns:
         sorted_data: a list of source data sorted by IDs from baseline_ids.txt file
    """

    n_labels = len(labels)
    sorted_data = list()

    for _ids in ids:
        if _ids in data:
            sorted_data.append(data[_ids])
        else:
            tmp = [0]*n_labels
            sorted_data.append(tmp)
    return sorted_data



def read_ids(path, ids_file_name, ids_colname, ids_has_header=True):
    """
    Function reads ids into the list
     
    Inputs:
         path: a string that defines a path to the directory the input data is stored
         ids_file_name: a string of ids file name
         ids_colname: a string of column name of ids
         ids_has_header: boolean if first column is a header
    Returns:
         ids: a list of personal identfiers (ID) from .txt ids file
    """
    # Setting header variable
    header=0 if ids_has_header else None
        
    # Reading the ids
    ids = pd.read_csv(path + ids_file_name, sep='\t', header=header)
    
    # Setting to column names and values to string
    ids = ids.astype('str')
    ids.columns = ids.columns.astype(str)
    
    ids = list(ids[str(ids_colname)])
    
    return ids


def read_files(path, data_type, na):
    """
    Function reads the input file into the dictionary
     
    Inputs:
         path: a string that defines a path to the directory the input data is stored
         data_type: a string that defines a name of .tsv file to encode
         na: a string that defines how NA values are defined in the source data file
    Returns:
         ids: a list of personal identfiers (ID) from baseline_ids.txt file
         raw_input: a dictionary with the data to encode
         header:a list of personal identfiers (ID) from .txt ids file
    """
                 
    raw_input = dict()
    with open(path + f"{data_type}.tsv", "r") as f:
        header = f.readline()
        for line in f:
            line = line.rstrip()
            tmp = np.array(line.split("\t"))
            vals = tmp[1:]
            vals[vals == na] = np.nan
            vals = list(map(float, vals))
            raw_input[tmp[0]] = vals
    header = header.split("\t")
     
    return raw_input, header

def generate_file(var_type, raw_data_path, interim_data_path, data_type, ids, na='NA'):
    """
    Function encodes source data type and saves the file
     
    inputs:
         var_type: a string out of ['categorical', 'continuous'], defines input data type to encode
         raw_data_path: a string that defines a path to the directory the input data is stored
         interim_data_path: a string that defines a path of directory where to save the files
         data_type: a string that defines a name of .tsv file to encode
         ids: a list of personal identfiers (ID) from .txt ids file
         na: a string that defines how NA values are defined in the source data file
    """
    
    isExist = os.path.exists(interim_data_path)
    if not isExist:
        os.makedirs(interim_data_path)
    
    # Preparing the data
    raw_input, header = read_files(raw_data_path, data_type, na)
    sorted_data = sort_data(raw_input, ids, header)
     
    if var_type == 'categorical':
        data_input = encode_cat(sorted_data, 'nan')
    elif var_type == 'continuous':
        data_input, _ = encode_con(sorted_data)
            
    np.save(interim_data_path + f"{data_type}.npy", data_input)

def get_list_value(*args):
    arg_tuple = [arg[0] if len(arg) == 1 else arg for arg in args]
    return(arg_tuple)     
