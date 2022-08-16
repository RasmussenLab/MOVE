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