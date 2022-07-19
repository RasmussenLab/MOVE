# Import functions
import hydra 
from move.conf.schema import MOVEConfig

from move.training.train import optimize_reconstruction
from move.utils.data_utils import get_data, merge_configs
from move.utils.visualization_utils import visualize_likelihood, visualize_recon_acc
from move.utils.analysis import make_and_save_best_reconstruct_params

@hydra.main(config_path="../conf", config_name="main")
def main(base_config: MOVEConfig): 
    
    # Overriding base_config with the user defined configs.
    cfg = merge_configs(base_config=base_config, 
                        config_types=['data', 'model', 'tuning_reconstruction'])
    
    # Getting the variables used in the notebook
    
    raw_data_path = cfg.data.raw_data_path
    interim_data_path = cfg.data.interim_data_path
    processed_data_path = cfg.data.processed_data_path
    data_of_interest = cfg.data.data_of_interest
    categorical_names = cfg.data.categorical_names
    continuous_names = cfg.data.continuous_names
    categorical_weights = cfg.data.categorical_weights
    continuous_weights = cfg.data.continuous_weights
    
    seed = cfg.model.seed
    cuda = cfg.model.cuda
    nepochs = cfg.model.num_epochs
    kld_steps = cfg.model.kld_steps
    batch_steps = cfg.model.batch_steps
    patience = cfg.model.patience
    lrate = cfg.model.lrate
    
    nHiddens = cfg.tuning_reconstruction.num_hidden
    nLatents = cfg.tuning_reconstruction.num_latent
    nLayers = cfg.tuning_reconstruction.num_layers
    nDropout = cfg.tuning_reconstruction.dropout
    nBeta = cfg.tuning_reconstruction.beta
    batch_sizes = cfg.tuning_reconstruction.batch_sizes
    repeat = cfg.tuning_reconstruction.repeats  
    max_param_combos_to_save = cfg.tuning_reconstruction.max_param_combos_to_save
    
    #Getting the data
    cat_list, con_list, cat_names, con_names, headers_all, drug, drug_h = get_data(raw_data_path, interim_data_path, categorical_names, continuous_names, data_of_interest)

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
        print('Visualizing the hyperparameter tuning results\n')
    except:
        print('Could not visualize the results\n')

    # Getting and saving the best n hyperparameter set value combinations for further optimisation 
    hyperparams_names = ['num_hidden','num_latent', 'num_layers', 'dropout', 'beta', 'batch_sizes']
    make_and_save_best_reconstruct_params(results_df, hyperparams_names, max_param_combos_to_save)

    return()


if __name__ == "__main__":
    main()