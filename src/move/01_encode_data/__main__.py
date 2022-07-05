# Import functions
import hydra 
from move.conf.schema import MOVEConfig
from move._utils.data_utils import generate_file


@hydra.main(config_path="../conf", config_name="main")
def main(config: MOVEConfig):
    
    # Define variables 
    path = config.data.processed_data_path
    ids_file_name = config.data.ids_file_name
    na_encoding = config.data.na_value
    categorical_names = config.model.categorical_names
    continuous_names = config.model.continuous_names
    
    # Encodes categorical data
    for cat_data in categorical_names:
        generate_file('categorical', path, cat_data, ids_file_name, na_encoding)
        print(f'Encoded {cat_data}')
    
    # Encodes continuous data 
    for con_data in continuous_names:
        generate_file('continuous', path, con_data, ids_file_name, na_encoding)    
        print(f'Encoded {con_data}')

if __name__ == "__main__":
    main()