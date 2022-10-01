# import hydra 
# from omegaconf import OmegaConf

# from move.conf.schema import MOVEConfig
# import docopt_subcommands as dsc
# import click


# @hydra.main(config_path="conf", config_name="main")
# def main(config: MOVEConfig) -> str:
#     print('''
#           To use MOVE pipeline, please run the following commands: 
#           python -m move.01_encode_data
#           python -m move.02_optimize_reconstruction
#           python -m move.03_optimize_stability
#           python -m move.04_analyze_latent
#           python -m move.05_identify_associations
          
#           To override the hyperparameter values, please write user defined values in the working directory in the following files:
#           data.yaml, model.yaml, tuning_reconstruction.yaml, tuning_stability, training_latent.yaml, training_association.yaml. 
#           '''
#          )
    
# if __name__ == "__main__":
#     main()

# Import functions
import logging

import os
import hydra 
import sys

from move.conf.schema import MOVEConfig
from move.utils.data_utils import read_ids, generate_file, merge_configs 
from move.utils.logger import get_logger

import datetime
import sys
from dataclasses import dataclass
from typing import Tuple, Union, Optional

import click
from omegaconf import OmegaConf, open_dict, read_write
import hydra
from hydra.core.config_store import ConfigStore
from hydra import initialize, compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
import copy

from move.modules.encode_data import main as f_encode_data 
from move.modules.optimize_reconstruction import main as f_optimize_reconstruction
from move.modules.optimize_stability import main as f_optimize_stability
from move.modules.analyze_latent import main as f_analyze_latent
from move.modules.identify_associations import main as f_identify_associations

###################################
# CLICK COMMAND GROUP DECLARATION
###################################

@click.group()
def cli():
    pass

# def merge_configs(base_config, config_types):
#     """
#     Merges base_config with user defined configuration
    
#     inputs:
#         base_config: YAML configuration
#         config_types: list of ints of names of user defined configuration types 
#     returns:
#         config_section: YAML configuration of base_config overrided by user defined configs and filtered for config_types classes
#     """
#     user_config_dict = dict()
#     override_types = []
    
#     # Getting the user defined configs 
#     for config_type in config_types:
#         exist = os.path.isfile(config_type + '.yaml')
#         if exist:    
#             override_types.append(config_type + '.yaml')
#             # Getting name of user config file and loading it 
#             user_config_name = base_config[config_type]['user_config']
#             user_config = OmegaConf.load(user_config_name)

#             # Making dict with the same key as in configuration file
#             user_config_dict[config_type] = user_config
    
#     # Printing what was overrided
#     override_types_str = ', '.join(str(override_type) for override_type in override_types)
    
#     logger.info(f'Overriding the default config with configs from {override_types_str}')
    
#     # Merging the base and user defined config file
#     config = OmegaConf.merge(base_config, user_config_dict)
    
#     # Getting a subsection of data used for printing 
#     config_section_dict = {x: config[x] for x in config_types if x in config}
#     config_section = OmegaConf.create(config_section_dict)

#     logger.info(f'\n\nConfiguration used:\n{OmegaConf.to_yaml(config_section)}')
#     return(config_section)

def store_schema() -> None:
    cs = ConfigStore.instance()
    cs.store(name="base_config_cs", node=MOVEConfig)

    
def merge_configs(base_config, **kwargs):
    """
    Merges base_config with user defined configuration
    
    inputs:
        base_config: YAML configuration
        config_types: list of ints of names of user defined configuration types 
    returns:
        config_section: YAML configuration of base_config overrided by user defined configs and filtered for config_types classes
    """
    print(kwargs.items())
    user_config_dict = dict()
    override_types = []
#     print(f'kwargs: {kwargs}')
    # Getting the user defined configs
#     print(base_config)
    with initialize(version_base="1.2", config_path="conf"):
        base_config = compose(config_name="main")
    print(f'base_config: {base_config}')
    config_types = []
#     print(kwargs.items())
    n_args = 1
    hydra_args = sys.argv[n_args + 1:]
    for key, value in kwargs.items():
        print(key)
#         print(key)
#         print(value)
#         assert os.path.isfile(value)
        ### Add if not NONE
#         print(base_config[key])
        user_config_name = base_config[key]['user_config']
        user_config = OmegaConf.load(user_config_name)

        # Making dict with the same key as in configuration file
        user_config_dict[key] = user_config
#         print(f'key: {key}')
#         print(OmegaConf.to_yaml(OmegaConf.create(user_config)))
        config_types.append(key)
        
#     for config_type in config_types:
#         exist = os.path.isfile(config_type + '.yaml')
#         if exist:    
#             override_types.append(config_type + '.yaml')
#             # Getting name of user config file and loading it 
#             user_config_name = base_config[config_type]['user_config']
#             user_config = OmegaConf.load(user_config_name)

#             # Making dict with the same key as in configuration file
#             user_config_dict[config_type] = user_config
    
    # Printing what was overrided
    override_types_str = ', '.join(str(override_type) for override_type in override_types)
    
#     logger.info(f'Overriding the default config with configs from {override_types_str}')
    
    # Merging the base and user defined config file
    config = OmegaConf.merge(base_config, user_config_dict)
    
    # Getting a subsection of data used for printing 
    config_section_dict = {x: config[x] for x in config_types if x in config}
    config_section = OmegaConf.create(config_section_dict)

#     logger.info(f'\n\nConfiguration used:\n{OmegaConf.to_yaml(config_section)}')
    return(config, hydra_args)

def remove_from_argv(**kwargs):
    '''
    Removing arguments used by click framework, keeping only the ones used by hydra.
    Strategy: if data file args are passed to the framework, they correspond to not NONE value, 
    it is multiplied by 2, since key ar vals are provided in sys.argv (e.g --data_config data.yaml).
    Finally, additionally removing one arg that corresponds to module name (e.g. encode-data)
    '''
    args_list = list(kwargs.values()) # Add removing specifically them
    n_args_rm = 2 * sum(x is not None for x in args_list) + 1
#     sys.argv = [sys.argv[0]] + sys.argv[n_args_rm + 1:]
    
    sys.argv = sys.argv[n_args_rm + 1:]
    return(sys.argv)
    print(f'sys.argv:{sys.argv}')

from dataclasses import dataclass

class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])
    
@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option('--data', type=click.Path(exists=True))
def encode_data(data):
#     hydra_args = remove_from_argv(**locals())
#     print(hydra_args)
    

            
            
#     print(hydra_args)
# #     cs = ConfigStore.instance()
# #     cs.store(name="main", group node=MOVEConfig)
# #     print(sys.argv)
# #     print(MOVEConfig)
#     cs = ConfigStore.instance()
#     cs.store(name="main", node=config_section)
#     with initialize(config_path="conf"):
#         cfg = compose(config_name="main", overrides=hydra_args)
#     f_encode_data(cfg)
    



#     cs = ConfigStore.instance()
    # Registering the Config class with the name 'config'.
    
#     cs.store(name="config", node=MOVEConfig)
#     from hydra import initialize, compose
#     with initialize(version_base=None, config_path="conf"):
#         base_config = compose(config_name="main")
#     print(f'locals: {locals().keys()}')
#     print(MOVEConfig)
#     with initialize(version_base="1.2", config_path="conf"):
#         config_section = compose(config_name="main")
        
#     print(f'config_section: {config_section}')
#     with open('best_first_configs.yaml', "w") as f:
#         OmegaConf.save(OmegaConf.create(dict(config_section)), f)
    
        
#     make_hydra_cfgs(config=config_section)    
        
        
#     HydraConfig.instance().set_config(config_section)
#     hydra_cfg = copy.deepcopy(HydraConfig.instance().cfg)
#     print(f'hydra_cfg: {hydra_cfg}')
#     task_cfg = copy.deepcopy(config_section)
    
#     with read_write(task_cfg):
#         with open_dict(task_cfg):
#             del task_cfg["hydra"]
        
    config_section, hydra_args = merge_configs(MOVEConfig, **locals())
    print(f'config_section: {config_section}')
#     print(f'localeee')
#     print(f'hydra_args: {hydra_args}')
#     print(hydra_args)
    store_schema() # https://stackoverflow.com/questions/70991020/how-to-reload-hydra-config-with-enumerations
#     print(config_section.keys())
    cs = ConfigStore.instance()
    cs.store(name="main", node=config_section)
    
#     cs.store(group='data', name="reload_config", node=config_section)
#     print('hydra_dir')
#     print(os.getcwd())
    exp_dir = os.path.abspath("outputs/2022-10-01/16-35-39/")
#    exp_dir = os.path.abspath('tmp3/')

    
    saved_cfg_dir = os.path.join(exp_dir, ".hydra")
    
#     cs.store(name="config2", node=MySQLConfig(host="test.db", port=3307))
    with initialize_config_dir(config_dir=saved_cfg_dir):
        base_config2 = compose(config_name="main", overrides=hydra_args)
    print(f'base_config2: {base_config2}')

@cli.command()
@click.option('--data_config', type=click.Path(exists=True))
@click.option('--model_config', type=click.Path(exists=True))
@click.option('--reconstruction_config', type=click.Path(exists=True))
def optimize_reconstruction(data_config, model_config, reconstruction_config):

    remove_from_argv(**locals())
    f_optimize_reconstruction()

    
@cli.command()
@click.option('--data_config', type=click.Path(exists=True))
@click.option('--model_config', type=click.Path(exists=True))
@click.option('--stability_config', type=click.Path(exists=True))
def optimize_stability(data_config, model_config, stability_config):

    remove_from_argv(**locals())
    f_optimize_stability()

    
@cli.command()
@click.option('--data_config', type=click.Path(exists=True))
@click.option('--model_config', type=click.Path(exists=True))
@click.option('--latent_config', type=click.Path(exists=True))
def analyze_latent(data_config, model_config, latent_config):

    remove_from_argv(**locals())
    f_analyze_latent()

    
@cli.command()
@click.option('--data_config', type=click.Path(exists=True))
@click.option('--model_config', type=click.Path(exists=True))
@click.option('--association_config', type=click.Path(exists=True))
def identify_associations(data_config, model_config, association_config):

    remove_from_argv(**locals())
    f_identify_associations()    

    
if __name__ == "__main__":
    cli()