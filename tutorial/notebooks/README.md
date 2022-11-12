## Pipeline

The pipeline consists of 5 steps (notebooks 01-05)

* The default hyperparameter values can be overridden by user-defined parameters stored in .yaml files in the working directory.
* Important: notebooks 2 and 3 return selected hyperparameter values into tuning_stability.yaml, training_latent.yaml and training_association.yaml overwriting it. 

#### Notebook 1 

* Takes hyperparameter values from data.yaml
* Processes raw input data into the .npy files ready to be taken by further steps of the pipeline
* Categorical data files are converted to one hot format, where nan values are set to zeros.
* Continuous data is log(x+1) transformed, then z-normalized. Features having all nan values are removed, and all other nan values are set to zeros.
* The feature names are saved in separate files.
* It is an example of data pre-processing. If needed, we recommend applying user-defined processing.

#### Notebook 2 

* Takes hyperparameter values from data.yaml, model.yaml and tuning_reconstruction.yaml
* Performs hyperparameter tuning for reconstruction for the hyperparameters of the number of hidden layers, number of nodes in hidden layers, latent size, probability of dropout after each nonlinearity, beta value (KLD weight coefficient).
* Selects the hyperparameter values that are among hyperparameter combinations having the highest reconstruction accuracy on the test set for further hyperparameter tuning
* Rounded mean number of best epochs of all the training is selected for all model training in other notebooks 
* Best hyperparameters are saved in tuning_stability.yaml file (it overwrites them!)
* We highly recommend reviewing the selected hyperparameter values and adjusting them if needed.

#### Notebook 3

* Takes hyperparameter values from data.yaml, model.yaml and tuning_stability.yaml
* Performs hyperparameter tuning for stability for the hyperparameters selected by notebook 2.
* Selects the hyperparameter set that shows the highest stability.
* Selects 4 latent sizes for the association analysis in notebook 5. 
* Saves the best hyperparameter values in training_latent.yaml and training_association.yaml (it overwrites them!)
* We highly recommend reviewing the selected hyperparameter values and adjusting them if needed (including by incorporating information from the results of notebook 2).

#### Notebook 4
* Takes hyperparameter values from data.yaml, model.yaml and training_latent.yaml
* Trains the model and explores its predictions: visualizes the reconstruction accuracy on the test set, visualizes how data points are distributed in the latent space, calculates Pearson correlations and measures feature importance.

#### Notebook 5

* Takes hyperparameter values from data.yaml, model. yaml and training_association.yaml
* Extracts drug (or selected categorical) data associations to all continuous datasets


## Hyperparameters

data.yaml:
```
na_value (str):  the string that corresponds to the NA value in the raw data files
raw_data_path (str): a pathway to the folder where raw data is located 
interim_data_path (str): a pathway to the folder where processed raw data will be saved (in .npy format)
results_path (str): a pathway to the folder where the results will be saved 
headers_path (str): pathway where the headers for interim_data will be saved
version (str): name of the subfolder in results_path where the results will be saved
ids_file_name (str): the name of the file that has data IDs (with the file suffix, e.g. baseline_ids.txt)
ids_has_header (boolean): if ids_file_name has header 
ids_colname (str): the name of the column where ids are stored (0 if ids_file_name has no header)
categorical_inputs (list): the list of names and weights of categorical type to use in the pipeline. 
  - name (str): name of file (the pipeline reads the raw_data_path + name + '.tsv')
    weight (int): weight of input type used in the pipeline
continuous_inputs (list): the list of names and weights of continuous type to use in the pipeline.
  - name: name of file (the pipeline reads the raw_data_path + name + '.tsv')
    weight: weight of input type used in the pipeline
data_of_interest (str): name of the data type whose effects on other data are analyzed by the pipeline  
data_features_to_visualize_notebook4 (list(str)): features to visualize in notebook 4  
write_omics_results_notebook5 (list(str)): data types to save results in notebook 5 
```

model.yaml:
```
seed (int): seed number
cuda (boolean): if using GPU for training the model
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
dropout (list(float)): the probability of dropout after each nonlinearity
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
dropout (list(float)): the probability of dropout after each nonlinearity
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
dropout (float): the probability of dropout after each nonlinearity
beta (float): KLD weight coefficient
batch_sizes (int): the size of batches during training
repeats (int):  times to repeat the training with each hyperparameter configuration
tuned_num_epochs (int): number of epochs to train the model (received in script 2)
```

training_association.yaml
```
num_hidden int): number of hidden nodes in hidden layers
num_latent (list(int)): dimension of latent space
num_layers int): number of hidden layers
dropout (float): the probability of dropout after each nonlinearity
beta (float): KLD weight coefficient
batch_sizes (int): the size of batches during training
repeats (int):  times to repeat the training with each hyperparameter configuration
tuned_num_epochs (int): number of epochs to train the model (received in script 2)
```