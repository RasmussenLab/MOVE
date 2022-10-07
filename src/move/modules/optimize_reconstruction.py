# Import functions
import hydra 
from move.conf.schema import MOVEConfig

from move.training.train import optimize_reconstruction
from move.utils.data_utils import get_data, merge_configs, make_and_save_best_reconstruct_params 
from move.utils.visualization_utils import visualize_likelihood, visualize_recon_acc
from move.utils.logger import get_logger

# @hydra.main(config_path="../conf", config_name="main", version_base="1.2")
# def main(base_config: MOVEConfig): 
def main(cfg, cfgs_save):     
    # Making logger for data writing
    logger = get_logger(logging_path='./logs/',
                        file_name='02_optimize_reconstruction.log',
                        script_name=__name__)
    
#     # Overriding base_config with the user defined configs.
#     cfg = merge_configs(base_config=base_config, 
#                         config_types=['data', 'model', 'tuning_reconstruction'])
    
    # Getting the variables used in the notebook

    interim_data_path = cfg.data_cfg.interim_data_path
    processed_data_path = cfg.data_cfg.processed_data_path
    headers_path = cfg.data_cfg.headers_path
    
    data_of_interest = cfg.data_cfg.data_of_interest
    categorical_names = cfg.data_cfg.categorical_names
    continuous_names = cfg.data_cfg.continuous_names
    categorical_weights = cfg.data_cfg.categorical_weights
    continuous_weights = cfg.data_cfg.continuous_weights
    
    seed = cfg.model_cfg.seed
    cuda = cfg.model_cfg.cuda
    nepochs = cfg.model_cfg.num_epochs
    kld_steps = cfg.model_cfg.kld_steps
    batch_steps = cfg.model_cfg.batch_steps
    patience = cfg.model_cfg.patience
    lrate = cfg.model_cfg.lrate
    
    nHiddens = cfg.reconstruction_cfg.num_hidden
    nLatents = cfg.reconstruction_cfg.num_latent
    nLayers = cfg.reconstruction_cfg.num_layers
    nDropout = cfg.reconstruction_cfg.dropout
    nBeta = cfg.reconstruction_cfg.beta
    batch_sizes = cfg.reconstruction_cfg.batch_sizes
    repeat = cfg.reconstruction_cfg.repeats  
    max_param_combos_to_save = cfg.reconstruction_cfg.max_param_combos_to_save
    
    #Getting the data
    cat_list, con_list, cat_names, con_names, headers_all, drug, drug_h = get_data(headers_path, interim_data_path, categorical_names, continuous_names, data_of_interest)

    #Performing hyperparameter tuning

    likelihood_tests, recon_acc_tests, recon_acc, results_df = optimize_reconstruction(nHiddens, nLatents, 
                                                                                       nLayers, nDropout, 
                                                                                       nBeta, batch_sizes, 
                                                                                       nepochs, repeat, 
                                                                                       lrate, kld_steps, 
                                                                                       batch_steps, patience, 
                                                                                       cuda, processed_data_path, 
                                                                                       cat_list, con_list,
                                                                                       continuous_weights, 
                                                                                       categorical_weights,
                                                                                       seed)
    

    # Visualizing the data
    try:
        visualize_likelihood(processed_data_path, nLayers, nHiddens, nDropout, nBeta, nLatents, likelihood_tests)
        visualize_recon_acc(processed_data_path, nLayers, nHiddens, nDropout, nBeta, nLatents, recon_acc_tests, 'test')
        visualize_recon_acc(processed_data_path, nLayers, nHiddens, nDropout, nBeta, nLatents, recon_acc, 'train')  
        logger.info('Visualizing the hyperparameter tuning results\n')
    except:
        logger.warning('Could not visualize the results\n')

    # Getting and saving the best n hyperparameter set value combinations for further optimisation 
    hyperparams_names = ['num_hidden','num_latent', 'num_layers', 'dropout', 'beta', 'batch_sizes']
    make_and_save_best_reconstruct_params(results_df, hyperparams_names, max_param_combos_to_save, cfgs_save)

    return()


