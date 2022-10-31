import os 
import numpy as np
import pandas as pd
import itertools
from collections import defaultdict
from omegaconf import OmegaConf
import logging

# Defining a logger name for logging
logger = logging.getLogger('data_utils')

def merge_configs(base_config, config_types):
    """ 
    Merges base_config with user defined configuration

    Args:
        base_config (omegaconf.dictconfig.DictConfig): YAML configuration
        config_types (list[int]): names of user defined configuration types

    Returns:
        omegaconf.dictconfig.DictConfig: YAML configuration of base_config overrided by user defined configs and filtered for config_types classes

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
    
    logger.info(f'Overriding the default config with configs from {override_types_str}')
    
    # Merging the base and user defined config file
    config = OmegaConf.merge(base_config, user_config_dict)
    
    # Getting a subsection of data used for printing 
    config_section_dict = {x: config[x] for x in config_types if x in config}
    config_section = OmegaConf.create(config_section_dict)

    logger.info(f'\n\nConfiguration used:\n{OmegaConf.to_yaml(config_section)}')
    return(config_section)


# Functions for loading data
def read_cat(path, file_name):
    """
    Reads categorical data file into numpy array

    Args:
        path (str): pathway to the directory where file is located
        file_name (str): file name to read (in .npy format) 

    Returns:
        np.array: input categorical data
    """    
    
    data = np.load(path + file_name)
    data = data.astype(np.float32)

    return data

def read_con(path, file_name):
    """
    Reads continuous data file into np.array, sets nan values as zeros and filters columns if all of the values were nan 
    
    Args:
        path (str): pathway to the directory where file is located
        file_name (str): file name to read (in .npy format)

    Returns:
        (tuple): tuple containing:
            data (np.array): of input data
            mask_col(np.array): np.array of Boolean values where False value corresponds to features that were filtered out
    """    
    data = np.load(path + file_name)
    data = data.astype(np.float32)
    data[np.isnan(data)] = 0
    consum = np.absolute(data).sum(axis=0)
    mask_col = consum != 0
    data = data[:,mask_col]
    return data, mask_col

def read_header(path, file_name, mask=None):
    """
    Reads features names from the headers

    Args:
        path (str): pathway to the directory where file is located
        file_name (str): file name to read (in .npy format)
        mask (np.array, optional): Bolean values that correspond to used features of input data. Defaults to None.

    Returns:
        list[str]: features names in the header
    """      

    header = pd.read_csv(path + file_name, sep='\t', header=None)
    header = header.squeeze().astype('str')

    if not mask is None:
        header =  header.to_numpy()
        header = header[mask]
        
    header = list(header)
    return header

def initiate_default_dicts(n_empty_dicts, n_list_dicts):
    """
    Initiates empty default dictionaries

    Args:
        n_empty_dicts (int): the number of how many defaultdicts without specified data type to initiate
        n_list_dicts (int): the number of how many defaultdicts with list type to initiate
    
    Returns:
        tuple: a tuple with initiated defaultdicts

    """    
    default_dicts = [defaultdict() for _ in range(n_empty_dicts)] + \
            [defaultdict(list) for _ in range(n_list_dicts)]
     
    return(tuple(default_dicts))

def get_data(headers_path, interim_data_path, categorical_data_names, continuous_data_names, data_of_interest):
    """
    Reads the data for models' inputs

    Args:
        headers_path (str): a pathway to headers data
        interim_data_path (str): a pathway to a folder of intermediate data files (e.g. .npy)
        categorical_data_names (list[str]): list of strings of categorical data type names
        continuous_data_names (list[str]): list of strings of continuous data type names
        data_of_interest (str): data type name whose features are changed to test their effects in the pipeline

    Returns:
        (tuple): a tuple containing:
            cat_list (list[np.array]): list of np.arrays for data of categorical data type 
            con_list (list[np.array]): list of np.arrays for data of continuous data type
            cat_names (list[np.array]): np.array of strings of feature names of categorical data
            con_names (list[np.array]): np.array of strings of feature names of continuous data
            headers_all (list[np.array]): np.array of strings of feature names of all data
            drug (np.array): np.array of input data whose feature data are changed to test their effects in the pipeline
            drug_h (np.array): np.array of strings of feature names data type whose data are changed to test their effects in the pipeline

    Raises:
        ValueError: In data.yaml file data_of_interest is chosen neither from defined continuous nor categorical data types
    """  
        
    # Initiate lists
    cat_list, cat_names, con_list, con_names = [], [], [], []
     
    # Get categorical variables
    for cat_data in categorical_data_names:
        cat = read_cat(interim_data_path, f"{cat_data}.npy")
        cat_h = read_header(headers_path,  f"{cat_data}.txt")
        cat_list.append(cat)
        cat_names.append(cat_h)

     # Get continuous variables
    for con_data in continuous_data_names:
        con, con_mask = read_con(interim_data_path, f"{con_data}.npy")
        con_h = read_header(headers_path, f"{con_data}.txt", con_mask)
          
        con_list.append(con)
        con_names.append(con_h)
 
    #Change data types
    headers_all = tuple(cat_names+con_names)
    con_names = np.concatenate(con_names)
    cat_names = np.concatenate(cat_names)

    # Select dataset of interest
    if data_of_interest in categorical_data_names:
        drug = read_cat(interim_data_path, f"{data_of_interest}.npy")
        drug_h = read_header(headers_path, f"{data_of_interest}.txt")
    elif data_of_interest in continuous_data_names:
        drug, _ = read_con(interim_data_path, f"{data_of_interest}.npy")
        drug_h = read_header(headers_path, f"{data_of_interest}.txt")  
    else:
        raise ValueError("""In data.yaml file data_of_interest is chosen neither
                                 from defined continuous nor categorical data types""")
     
    return(cat_list, con_list, cat_names, con_names, headers_all, drug, drug_h)


# Functions for encoding

def encode_cat(sorted_data, na='NA'):
    """
    Encodes categorical data into one-hot encoding

    Args:
        sorted_data (list): a list of source data sorted by IDs from baseline_ids.txt file
        na (str, optional): the string how NA values are encoded in raw input data. Defaults to 'NA'.

    Returns:
        np.array: one hot encoded data
    """ 
     
    matrix = np.array(sorted_data)
    n_labels = matrix.shape[1]
    n_samples = matrix.shape[0]
    
    unique_sorted_data = np.unique(sorted_data)
    x = np.where(unique_sorted_data == 'nan')
    unique_sorted_data_nonan = np.delete(unique_sorted_data, x)
    num_classes = len(unique_sorted_data_nonan)
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

def encode_con(sorted_data):
    """
    Log transforms and z-normalizes the data

    Args:
        sorted_data (list): a list of source data sorted by IDs from baseline_ids.txt file

    Returns:
        (tuple): a tuple containing:
            data_input (np.array): numpy array with log transformed and z-score normalized data
            mask_col (np.array): a np.array vector of Bolean values that correspond to nonzero sd values    
    """
    
    matrix = np.array(sorted_data)
    consum = np.nansum(matrix, axis=1)
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
    
    # change all nan in input to zero for encoding
    np_index = np.isnan(data_input)
    data_input[np_index] = 0
    
    return data_input, mask_col 


def sort_data(data, ids, labels):
    """
    Sorts data based on the ids file

    Args:
        data (dict): a dictionary with the data to encode
        ids (list): a list of personal identfiers (ID) from baseline_ids.txt file
        labels (list): a list of column names from the source data file

    Returns:
        list: a list of source data sorted by IDs from baseline_ids.txt file
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

    Args:
        path (str): a string that defines a path to the directory the input data is stored
        ids_file_name (str): a string of ids file name
        ids_colname (str): a string of column name of ids
        ids_has_header (bool, Bol): boolean if first column is a header. Defaults to True.

    Returns:
        list: a list of personal identfiers (ID) from .txt ids file
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


def read_files(var_type, path, data_type, na):
    """
    Function reads the input file into the dictionary

    Args:
        var_type (str): a string out of ['categorical', 'continuous'], defines input data type to encode
        path (str): a string that defines a path to the directory the input data is stored
        data_type (str): a string that defines a name of .tsv file to encode
        na (str): a string that defines how NA values are defined in the raw input data file

    Returns:
        (tuple): a tuple containing:
            ids (list): a list of personal identfiers (ID) from baseline_ids.txt file
            raw_input (dict): a dictionary with the data to encode
            header (list): a list of personal identfiers (ID) from .txt ids file
    """  
                 
    raw_input = dict()
    with open(path + f"{data_type}.tsv", "r") as f:
        header = f.readline()
        for line in f:
            line = line.rstrip()
            # make sure to enforce 25-char string to make sure no problems with missing
            # when converting from e.g. NA to nan and it reads automatically as <U2
            tmp = np.array(line.split("\t"), dtype=np.dtype('U25'))
            vals = tmp[1:]
            if var_type == 'categorical':
                # store indexes of NAs
                na_index = np.where(vals==na)
                vals[na_index] = np.nan
                raw_input[tmp[0]] = vals
            elif var_type == 'continuous':
                na_index = np.where(vals==na)
                vals[na_index] = np.nan
                vals_fl = list(map(float, vals))
                raw_input[tmp[0]] = vals_fl
    header = header.split("\t")
    
    return raw_input, header

def generate_file(var_type, raw_data_path, interim_data_path, headers_path, data_type, ids, na='NA'):
    """
    Function encodes source data type and saves the file

    Args:
        var_type (str): a string out of ['categorical', 'continuous'], defines input data type to encode
        raw_data_path (str): a string that defines a path to the directory the input data is stored
        interim_data_path (str): a string that defines a path of directory where to save the files
        headers_path (str): a string that defines a path of directory where to save headers
        data_type (str): a string that defines a name of .tsv file to encode
        ids (list): a list of personal identfiers (ID) from .txt ids file
        na (str, optional): a string that defines how NA values are defined in the raw input data file. Defaults to 'NA'.
    """    
    
    isExist = os.path.exists(interim_data_path)
    if not isExist:
        os.makedirs(interim_data_path)
    
    isExist = os.path.exists(headers_path)
    if not isExist:
        os.makedirs(headers_path)
    
    # Preparing the data
    raw_input, header = read_files(var_type, raw_data_path, data_type, na)
    sorted_data = sort_data(raw_input, ids, header)
    
    if var_type == 'categorical':
        data_input = encode_cat(sorted_data, 'nan')
        mask = None
    
    elif var_type == 'continuous':
        data_input, mask = encode_con(sorted_data)
    
    header = get_header(header, mask)
    
    np.savetxt(headers_path+data_type+'.txt', header, delimiter=',', fmt='%s')
    np.save(interim_data_path + f"{data_type}.npy", data_input)
    
    logger.info(f'  Encoded {data_type}')

def get_header(header, mask=None, start=1):
    """
    Reads features names from the headers

    Args:
        header (list): list with values of the header column of input data
        mask (np.array, optional): np.array of boolean objects, where False value corresponds to features to filtered out. Defaults to None.
        start (int, optional): number corresponding to how many lines to read for the header. Defaults to 1.

    Returns:
        np.array: np.array of strings of feature names
    """     
    header = header[start:]
    if not mask is None:
        header = np.array(header)
        header = header[mask]
    
    return header    

def get_best_epoch(results_df):
    """
    Gets the number of epochs for further experiments after hyperparameter tuning for reconstruction 

    Args:
        results_df (pd.DataFrame): pd.DataFrame of results after hyperparameter tuning for reconstruction 
    
    Returns:
        (int): number of epochs used in further computations

    """    
    #Rounding best_epoch to closest 10 (just in case if best_epoch is less than 5 - to closest number)
    round_epoch = lambda x : round(x, 0) if (x < 5) else round(x, -1)
    best_epoch = results_df['best_epochs'].mean()
    best_epoch = int(round_epoch(best_epoch))
    
    return(best_epoch)


def get_sort_list(results_df):
    """
    Sorts DataFrame by mean recustruction accuracy on test set

    Args:
        results_df (pd.DataFrame): pd.DataFrame with results of hyperparameter tuning for reconstruction 

    Returns:
        pd.DataFrame: sorted pd.DataFrame by mean recustruction accuracy on test set
    """ 
    #Gets mean values of test accuracy reconstruction
    results_df['recon_acc_test_mean'] = results_df['recon_acc_test'].map(lambda x: x.mean())
    
    # Sort values by reconstruction accuracy
    results_df = results_df.sort_values('recon_acc_test_mean', ascending=False)
    return (results_df)


def get_length(hyperpars_vals_dict, hyperpar_name):
    """
    gets the number of total combinations with given hyperparameter value set

    Args:
        hyperpars_vals_dict (defaultdict(list)): defaultdict with hyperparameter keys and values of hyperparameters that gave best mean recustruction accuracy on test set
        hyperpar_name (str): name of hyperparameter

    Returns:
        (int): the number of total combinations with given hyperparameter value set
    """  
    lengths = 1
    for key in hyperpars_vals_dict:  
        length = len(hyperpars_vals_dict[key])
        # Adding +1 since we want to check if adding a parameter it overreaches
        if key == hyperpar_name:
            length+=1
        lengths *= length

    return(lengths)

def get_best_params(results_df_sorted, n_combos_opt, hyperpars_names):
    """
    Gets a set of hyperparameter values of no more than n_combos_opt combinations that showed the highest mean reconstruction accuracy 

    Args:
        results_df_sorted (pd.DataFrame): sorted pd.DataFrame by mean recustruction accuracy on test set
        n_combos_opt (int): number of maximum total of combinations to save for further optimization
        hyperpars_names (list[str]): list of hyperparameter values that are used in hyperparameter optimization

    Returns:
        (dict): dict with best hyperparameter values used in further optimization 
    """ 
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
    """
    Gets the dictionary with best hyperparameter values used in further optimization and save these results

    Args:
        results_df (pd.DataFrame): pd.DataFrame with results of hyperparameter tuning for reconstruction 
        hyperparams_names (list[str]): list of hyperparameter values that are used in hyperparameter optimization
        max_param_combos_to_save (int): number of maximum total of combinations to save for further optimization
    """  
    # Getting the best number of epochs used in further trainings
    best_epoch = get_best_epoch(results_df)
    
    logger.info('Starting calculating the best hyperparameter values for further optimization') 
    
    results_df_sorted = get_sort_list(results_df)
    best_hyperpars_vals_dict = get_best_params(results_df_sorted, max_param_combos_to_save, hyperparams_names)
    best_hyperpars_vals_dict['tuned_num_epochs'] = best_epoch
    
    # Saving the best hyperparameter values (it will overwrite the file if it already exists)
    with open('tuning_stability.yaml', "w") as f:
        OmegaConf.save(OmegaConf.create(dict(best_hyperpars_vals_dict)), f)

    #Printing the saved hyper parameter values
    logger.info(f'Saving the best hyperparameter values in tuning_stability.yaml for further optimization:\n \n{OmegaConf.to_yaml(dict(best_hyperpars_vals_dict))}\n')
    logger.warning('Please manually review if the hyperparameter values were selected correctly and adjust them in the tuning_stability.yaml file.')
    

def get_best_stability_paramset(stability_df, hyperparams_names):
    """
    Gets the hyperparameters values of the lowest difference among reconstructions

    Args:
        stability_df (pd.DataFrame): pd.DataFrame with results of hyperparameter tuning for stability
        hyperparams_names (list[str]): list of hyperparameter values that are used in hyperparameter optimization
    
    Returns:
        (tuple): tuple containing:
            params_to_save (dict): dictionary of parameters to save for further optimization
            stability_df_sorted (pd.DataFrame): pd.DataFrame with results of hyperparameter tuning for stability sorted by mean difference among reconstruction
    """ 
    params_to_save = dict()
    stability_df_sorted = stability_df.sort_values('difference', ascending=False).iloc[:1]
    for hyperparam in hyperparams_names: 
        params_to_save[hyperparam] = stability_df_sorted[hyperparam].item()
        
    return(params_to_save, stability_df_sorted)

def get_best_4_latent_spaces(results_df_sorted):
    """
    Gets best four latent space sizes used for the script 5

    Args:
        results_df_sorted (pd.DataFrame): pd.DataFrame with results of hyperparameter tuning for stability sorted by mean difference among reconstruction

    Returns:
        (list): list of ints with best 4 latent spaces sizes
    """    
    #Selecting best two latent spaces
    best_latent = []
    for index, row in results_df_sorted.iterrows():
        if row['num_latent'] not in best_latent:
            best_latent.append(int(row['num_latent']))
        if len(best_latent) >= 2:
            break
            
    # Finding difference between best 2 latent spaces, and half size of the lowest best latent size
    best_hypers_diff = max(best_latent) - min(best_latent)
    half_diff_from_zero = int(min(best_latent)/2)
    
    # If only one latent space exist among user defined hyperparameter values, 
    # appending 0.5, 1.5 and 2 sizes of the latent space.
    if best_hypers_diff == 0:
        best_latent.append(min(best_latent) - half_diff_from_zero)
        best_latent.append(max(best_latent) + half_diff_from_zero)        
        best_latent.append(max(best_latent) + half_diff_from_zero) 
    # Else if the difference between best latent sizes is lower than half size of the lowest best latent size
    # subtracting it from lowest latent size value, and adding it the highest value, and appending both of them among latent spaces   
    elif best_hypers_diff < half_diff_from_zero:
        best_latent.append(min(best_latent) - best_hypers_diff)
        best_latent.append(max(best_latent) + best_hypers_diff)
    # Else adding and subtracting half of the lowest latent space to lowest and biggest best latent space sizes 
    else:
        best_latent.append(half_diff_from_zero)
        best_latent.append(max(best_latent) + half_diff_from_zero)
    return(best_latent)

def make_and_save_best_stability_params(results_df, hyperparams_names, nepochs):
    """
    Gets the dictionaries with best hyperparameter values used in further optimization and save these results

    Args:
        results_df (pd.DataFrame): pd.DataFrame with results of hyperparameter tuning for stability
        hyperparams_names (list[str]): list of hyperparameter values that are used in hyperparameter optimization
        nepochs (int): best number of epochs that was calulated in the 2nd script
    """    
    logger.info('Starting calculating the best hyperparameter values used in further model trainings') 
    
    # Getting best set of hyperparameters
    params_to_save, results_df_sorted = get_best_stability_paramset(results_df, hyperparams_names)
    params_to_save['tuned_num_epochs'] = nepochs

    # Saving best set of hyperparameters    
    with open('training_latent.yaml', "w") as f:
        OmegaConf.save(OmegaConf.create(dict(params_to_save)), f)
        
    # Printing the configuration saved 
    logger.info(f'Saving best hyperparameter values in training_latent.yaml: \n\n{OmegaConf.to_yaml(dict(params_to_save))}')
    
    # Getting the latent spaces for training_association script and using them with the best hyperparam set
    best_latent = get_best_4_latent_spaces(results_df_sorted)
    params_to_save['num_latent'] = list(best_latent)
    
    # Saving best set of hyperparameters for training_association script
    with open('training_association.yaml', "w") as f:
        OmegaConf.save(OmegaConf.create(dict(params_to_save)), f)

    # Printing the configuration saved 
    logger.info(f'Saving best hyperparameter values in training_association.yaml: \n \n{OmegaConf.to_yaml(dict(params_to_save))}')
    logger.warning('Please manually review if the hyperparameter values were selected correctly and adjust them in the training_association.yaml and training_latent.yaml files.')

def read_saved_files(nLatents, repeats, path, version):
    """
    Reads the saved files of 5th script 

    Args:
        nLatents (list): list of latent sizes used in running 5th script 
        repeats (int): number of repeats used in running 5th script
        path (str): path to the folder where results were saved 
        version (str): version where the results were saved
    
    Returns:
        (tuple): tuple containing:
            results (dict): TODO
            recon_results (dict): {latents: {repeat: {drug: np.array of changes in continuous data when label of drug is changed}}}
            groups (dict): TODO
            mean_bas (dict): dict with keys as nLatent and values of np.arrays of floats with means of reconstruction differences between forwards of network
    """   
    results, recon_results, groups, mean_bas = initiate_default_dicts(n_empty_dicts=0, n_list_dicts=4)
    
    iters = itertools.product(nLatents, range(repeats))
    for nLatent, repeat in iters:
        result = np.load(path + f'05_identify_associations/results_{str(nLatent)}_{str(repeat)}_{version}.npy', mmap_mode='r', allow_pickle = True)
        mean_ba = np.load(path + f'05_identify_associations/mean_bas_{str(nLatent)}_{str(repeat)}_{version}.npy', mmap_mode='r', allow_pickle = True)
        recon_result = np.load(path + f'05_identify_associations/recon_results_{str(nLatent)}_{str(repeat)}_{version}.npy', mmap_mode='r', allow_pickle = True)
        recon_result_dict = {i:recon_result[i] for i in range(recon_result.shape[0])}
        
        results[nLatent].append(result)
        recon_results[nLatent].append(recon_result_dict)
        mean_bas[nLatent].append(mean_ba) 
    
    groups = np.load(path + "05_identify_associations/results_groups_" + version + ".npy", mmap_mode='r', allow_pickle = True)
    groups_dict = {i:groups[i] for i in range(groups.shape[0])}
    
    return(results, recon_results, groups_dict, mean_bas)
