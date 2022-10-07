# Import functions
import logging

import os
import hydra 
import sys
from omegaconf import OmegaConf

from move.conf.schema import MOVEConfig
from move.utils.data_utils import read_ids, generate_file#, merge_configs 
from move.utils.logger import get_logger


# @hydra.main(config_path="../conf", config_name="main", version_base="1.2")
def main(cfg):
    
    # Making logger for data writing
    logger = get_logger(logging_path='./logs/',
                        file_name='01_encode_data.log',
                        script_name=__name__)
    
    logger.info(f'\n\nConfiguration used:\n{OmegaConf.to_yaml(cfg)}')
    
#     # Overriding base_config with the user defined configs.
#     cfg = merge_configs(base_config=base_config, 
#                         config_types=['data'])

    # Getting the variables used in the notebook
    raw_data_path = cfg.data_cfg.raw_data_path
    interim_data_path = cfg.data_cfg.interim_data_path
    headers_path = cfg.data_cfg.headers_path
    ids_file_name = cfg.data_cfg.ids_file_name
    ids_has_header = cfg.data_cfg.ids_has_header
    ids_colname = cfg.data_cfg.ids_colname
    
    na_encoding = cfg.data_cfg.na_value
    categorical_names = cfg.data_cfg.categorical_names
    continuous_names = cfg.data_cfg.continuous_names    
    
    # Reading ids 
    ids = read_ids(raw_data_path, ids_file_name, ids_colname, ids_has_header)

    # Encoding categorical data
    logger.info('Encoding categorical data')
    for cat_data in categorical_names:
        generate_file('categorical', raw_data_path, interim_data_path, headers_path, cat_data, ids, na_encoding)
    
    # Encoding continuous data 
    logger.info('Encoding continuous data')
    for con_data in continuous_names:
        generate_file('continuous', raw_data_path, interim_data_path, headers_path, con_data, ids, na_encoding)   
        

