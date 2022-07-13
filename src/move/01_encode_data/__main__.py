# Import functions
import hydra 
from move.conf.schema import MOVEConfig

from move.utils.data_utils import generate_file, merge_configs 

@hydra.main(config_path="../conf", config_name="main")
def main(base_config: MOVEConfig):
    
    # Merging the user defined data.yaml, model.yaml and tuning_reconstruction.yaml 
    # with the base_config to override it.
    print('Overriding the default configuration with configuration from data.yaml')
    cfg = merge_configs(base_config=base_config, 
                        config_types=['data'])
    
    # Getting the variables used in the notebook
    path = cfg.data.processed_data_path
    ids_file_name = cfg.data.ids_file_name
    na_encoding = cfg.data.na_value
    categorical_names = cfg.data.categorical_names
    continuous_names = cfg.data.continuous_names    
    
    # Encoding categorical data
    print('Encoding categorical data')
    for cat_data in categorical_names:
        generate_file('categorical', path, cat_data, ids_file_name, na_encoding)
        print(f'  Encoded {cat_data}')
    
    # Encoding continuous data 
    print('Encoding continuous data')
    for con_data in continuous_names:
        generate_file('continuous', path, con_data, ids_file_name, na_encoding)    
        print(f'  Encoded {con_data}')

if __name__ == "__main__":
    main()
