# Import functions
import hydra 
from move.conf.schema import MOVEConfig

from move.utils.data_utils import read_ids, generate_file, merge_configs 

@hydra.main(config_path="../conf", config_name="main")
def main(base_config: MOVEConfig):
    
    # Overriding base_config with the user defined configs.
    cfg = merge_configs(base_config=base_config, 
                        config_types=['data'])
    
    # Getting the variables used in the notebook
    raw_data_path = cfg.data.raw_data_path
    interim_data_path = cfg.data.interim_data_path
    ids_file_name = cfg.data.ids_file_name
    ids_has_header = cfg.data.ids_has_header
    ids_colname = cfg.data.ids_colname
    
    na_encoding = cfg.data.na_value
    categorical_names = cfg.data.categorical_names
    continuous_names = cfg.data.continuous_names    
    
    # Reading ids 
    ids = read_ids(raw_data_path, ids_file_name, ids_colname, ids_has_header)

    # Encoding categorical data
    print('Encoding categorical data')
    for cat_data in categorical_names:
        generate_file('categorical', raw_data_path, interim_data_path, cat_data, ids, na_encoding)
        print(f'  Encoded {cat_data}')
    
    # Encoding continuous data 
    print('Encoding continuous data')
    for con_data in continuous_names:
        generate_file('continuous', raw_data_path, interim_data_path, con_data, ids, na_encoding)    
        print(f'  Encoded {con_data}')

if __name__ == "__main__":
    main()