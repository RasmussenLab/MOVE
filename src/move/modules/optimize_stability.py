# Import functions
import hydra 
from move.conf.schema import MOVEConfig

from move.training.train import optimize_stability
from move.utils.data_utils import get_data, get_list_value, merge_configs, make_and_save_best_stability_params
from move.utils.visualization_utils import draw_boxplot
from move.utils.analysis import get_top10_stability, calculate_latent
from move.utils.logger import get_logger

# @hydra.main(config_path="../conf", config_name="main", version_base="1.2")
def main(cfg, cfgs_save): 
    # Making logger for data writing
    logger = get_logger(logging_path='./logs/',
                        file_name='03_optimize_stability.log',
                        script_name=__name__)
    
#     # Overriding base_config with the user defined configs.
#     cfg = merge_configs(base_config=base_config, 
#                         config_types=['data', 'model', 'tuning_stability'])
    
    #Getting the variables used in the notebook
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
    lrate = cfg.model_cfg.lrate
    kld_steps = cfg.model_cfg.kld_steps
    batch_steps = cfg.model_cfg.batch_steps
    
    nHiddens = cfg.stability_cfg.num_hidden
    nLatents = cfg.stability_cfg.num_latent
    nLayers = cfg.stability_cfg.num_layers
    nDropout = cfg.stability_cfg.dropout
    nBeta = cfg.stability_cfg.beta
    batch_sizes = cfg.stability_cfg.batch_sizes
    repeat = cfg.stability_cfg.repeats
    nepochs = cfg.stability_cfg.tuned_num_epochs
    
    # Raising the error if more than 1 batch size is used 
    if len(batch_sizes)==1:
        batch_sizes = batch_sizes[0]
    elif len(batch_sizes)>1:
        raise('Currently the code is implemented to take take only one value for batch_size')
    
    #Getting the data
    cat_list, con_list, cat_names, con_names, headers_all, drug, drug_h = get_data(headers_path, interim_data_path, categorical_names, continuous_names, data_of_interest)
    
    #Performing hyperparameter tuning
    embeddings, latents, con_recons, cat_recons, recon_acc = optimize_stability(nHiddens, nLatents, 
                                                                                nDropout, nBeta, repeat,
                                                                                nepochs, nLayers,
                                                                                batch_sizes, lrate, 
                                                                                kld_steps, batch_steps, 
                                                                                cuda, processed_data_path, 
                                                                                con_list, cat_list,
                                                                                continuous_weights, categorical_weights,
                                                                                seed)
    
    # Getting stability results 
    stability_top10, stability_top10_df = get_top10_stability(nHiddens, nLatents, nDropout, nLayers, repeat, latents, batch_sizes, nBeta)
    
    stability_total, rand_index, stability_total_df = calculate_latent(nHiddens, nLatents, nDropout, repeat, nLayers, nBeta, latents, batch_sizes)
 
    
    # Plotting the results 
    try: 
        draw_boxplot(path=processed_data_path ,
                     df=stability_top10,
                     title_text='Difference across replicationes in cosine similarity of ten closest neighbours in first iteration',
                     y_label_text="Average change",
                     save_fig_name="stability_top10")

        draw_boxplot(path=processed_data_path ,
                     df=stability_total,
                     title_text='Difference across replicationes in cosine similarity compared to first iteration',
                     y_label_text="Average change",
                     save_fig_name="stability_all")

        draw_boxplot(path=processed_data_path,
                     df=rand_index,
                     title_text='Rand index across replicationes compared to first iteration',
                     y_label_text="Rand index",
                     save_fig_name="rand_index_all")
        logger.info('Visualizing the hyperparameter tuning results\n')
        
    except:
        logger.warning('Could not visualize the results\n')
    
    # Getting best set of hyperparameters
    hyperparams_names = ['num_hidden', 'num_latent', 'num_layers', 'dropout', 'beta', 'batch_sizes']
    make_and_save_best_stability_params(stability_total_df, hyperparams_names, nepochs, cfgs_save)

    return()

