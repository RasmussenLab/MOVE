# Import functions

import numpy as np
from collections import defaultdict
import pandas as pd
import yaml
from move._utils.data_utils import *
import os
print(os.getcwd())
import hydra 
from omegaconf import OmegaConf
from move.conf.schema import MOVEConfig


@hydra.main(config_path="../conf", config_name="main")
def main(config: MOVEConfig):
    # Define variables 
    path = config.data.processed_data_path
    ids_file_name = config.data.ids_file_name
    na_encoding = config.data.na_value

    # Encodes categorical data
    for cat_data in config.model.categorical_names:
        generate_file('categorical', path, cat_data, ids_file_name, na_encoding)
        print(f'Encoded {cat_data}')
    
    # Encodes continuous data 
    for con_data in config.model.continuous_names:
        generate_file('continuous', path, con_data, ids_file_name, na_encoding)    
        print(f'Encoded {con_data}')

        
if __name__ == "__main__":
    main()