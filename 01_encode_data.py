# Import functions

import numpy as np
from collections import defaultdict
import pandas as pd
import yaml
from utils.data_utils import *

if __name__ == "__main__":
    
    # Reads the data as dictionary
    data_dict = read_yaml('data')
    
    # Takes variables from the read file
    path = data_dict['path']
    ids_file_name = data_dict['ids_file_name']
    na_encoding = data_dict['na_encoding']

    # Encodes categorical data
    for cat_data in data_dict['categorical_data_files']:
        generate_file('categorical', path, 
                      cat_data, ids_file_name, na_encoding)
        print(f'Encoded {cat_data}')
    
    # Encodes continuous data 
    for con_data in data_dict['continuous_data_files']:
        generate_file('continuous', path, con_data, ids_file_name, na_encoding)    
        print(f'Encoded {con_data}')