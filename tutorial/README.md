## Parameters

data.yaml:
```
na_value (str):  the string that corresponds to NA value in the raw data files
raw_data_path (str): pathway to the folder where raw data is located 
interim_data_path (str): pathway to the folder where processed raw data will be saved (in .npy format)
processed_data_path (str): pathway to the folder where the results will be saved 
headers_path (str): pathway where the headers for interim_data will be saved
version (str): name of subfolder in processed_data_path where the results will be saved
ids_file_name (str): the name of file that has data IDs  
ids_has_header (boolean): if ids_file_name has header 
ids_colname (str): the name of column where ids are stored (0 if ids_file_name has no header)
categorical_inputs (list): the list of names and weights of categorical type to use in the pipeline. 
  - name (str): name of file (the pipeline reads the raw_data_path + name + '.tsv')
    weight (int): weight of input type used in the pipeline
continuous_inputs (list): the list of names and weights of continuous type to use in the pipeline.
  - name: name of file (the pipeline reads the raw_data_path + name + '.tsv')
    weight: weight of input type used in the pipeline
data_of_interest (str): name of the data type, which effect on other data is analyzed by the pipeline  
data_features_to_visualize_notebook4 (list(str)): features to visualize in noteboke 4  
write_omics_results_notebook5 (list(str)): data types to save rezults in notebook 5 
```

model.yaml:
```
seed (int): seed number
cuda (boolean): if use GPU for training the model
lrate (float): learning rate for model
num_epochs (int): number of epochs to train the model
patience (int): number of epochs without validation improvement before termination run
kld_steps (list(int)): epochs when KLD weight is increased
batch_steps (list(int)): epochs when batch size is increased
```

tuning_reconstruction.yaml
```
num_hidden (list(int)): number of hidden nodes in hidden layers
num_latent (list(int)): dimension of latent space
num_layers (list(int)): number of hidden layers
dropout (list(float)): probability of dropout after each nonlinearity
beta (list(float)): KLD weight coefficient
batch_sizes (list(int)): size of batches during training
repeats (int): times to repeat the training with each hyperparameter configuration
max_param_combos_to_save (int): maximum number of hyperparameter combinations to save for hyperparameter tuning for stability
```

tuning_stability.yaml
```
num_hidden (list(int)): number of hidden nodes in hidden layers
num_latent (list(int)): dimension of latent space
num_layers (list(int)): number of hidden layers
dropout (list(float)): probability of dropout after each nonlinearity
beta (list(float)): KLD weight coefficient
batch_sizes (list(int)): size of batches during training
repeats (int):  times to repeat the training with each hyperparameter configuration
tuned_num_epochs (int): number of epochs to train the model (received by script or notebook 02)
```

training_latent.yaml
```
num_hidden (int): number of hidden nodes in hidden layers
num_latent (int): dimension of latent space
num_layers (int): number of hidden layers
dropout (float): probability of dropout after each nonlinearity
beta (float): KLD weight coefficient
batch_sizes (int): size of batches during training
repeats (int):  times to repeat the training with each hyperparameter configuration
tuned_num_epochs (int): number of epochs to train the model (received in script 2)
```

training_association.yaml
```
num_hidden int): number of hidden nodes in hidden layers
num_latent (list(int)): dimension of latent space
num_layers int): number of hidden layers
dropout (float): probability of dropout after each nonlinearity
beta (float): KLD weight coefficient
batch_sizes (int): size of batches during training
repeats (int):  times to repeat the training with each hyperparameter configuration
tuned_num_epochs (int): number of epochs to train the model (received in script 2)
```