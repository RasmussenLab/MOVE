# Load functions
import hydra 
from move.conf.schema import MOVEConfig

from move.training.train import train_model_association
from move.utils.data_utils import get_data, merge_configs
from move.utils.visualization_utils import visualize_indi_var, visualize_drug_similarity_across_all
from move.utils.analysis import cal_reconstruction_change, overlapping_hits, identify_high_supported_hits, report_values, get_change_in_reconstruction, write_omics_results, make_files, get_inter_drug_variation, get_drug_similar_each_omics

import numpy as np 

@hydra.main(config_path="../conf", config_name="main")
def main(base_config: MOVEConfig): 
    
    # Overriding base_config with the user defined configs.
    cfg = merge_configs(base_config=base_config, 
                        config_types=['data', 'model', 'training_association'])
    
    #Getting the variables used in the notebook
    interim_data_path = cfg.data.interim_data_path
    processed_data_path = cfg.data.processed_data_path 
    headers_path = cfg.data.headers_path
    
    data_of_interest = cfg.data.data_of_interest
    version = cfg.data.version
    categorical_names = cfg.data.categorical_names
    continuous_names = cfg.data.continuous_names
    categorical_weights = cfg.data.categorical_weights
    continuous_weights = cfg.data.continuous_weights
    up_down_list = cfg.data.write_omics_results_notebook5
    
    seed = cfg.model.seed
    cuda = cfg.model.cuda
    lrate = cfg.model.lrate
    kld_steps = cfg.model.kld_steps
    batch_steps = cfg.model.batch_steps

    nHiddens = cfg.training_association.num_hidden
    nLatents = cfg.training_association.num_latent
    nLayers = cfg.training_association.num_layers
    nDropout = cfg.training_association.dropout
    nBeta = cfg.training_association.beta
    batch_sizes = cfg.training_association.batch_sizes
    nepochs = cfg.training_association.tuned_num_epochs
    repeats = cfg.training_association.repeats
    
    types = [[1, 0]]
    
    # Checking if all data types selected for visualization are in continuous_names
    for data_type in up_down_list:
        if data_type not in continuous_names:
            raise ValueError(f"{data_type} is not in the continuous_names list.")
    
    # Getting the data
    cat_list, con_list, cat_names, con_names, headers_all, drug, drug_h = get_data(headers_path, interim_data_path, categorical_names, continuous_names, data_of_interest)
    
    # Training the model
    print('Beginning training the model.\n')
    train_model_association(processed_data_path, cuda, nepochs, nLatents, batch_sizes, nHiddens, nLayers, nBeta, nDropout, con_list, cat_list, continuous_weights, categorical_weights, version, repeats, kld_steps, batch_steps, lrate, drug, categorical_names, data_of_interest, seed)
    print('\nFinished training the model.')
    
    # Loading the saved files by train_model_association() - for using the results without the need to rerun the function
    results = np.load(processed_data_path + "results/results_" + version + ".npy", allow_pickle=True).item()
    recon_results = np.load(processed_data_path + "results/results_recon_" + version + ".npy", allow_pickle=True).item()
    groups = np.load(processed_data_path + "results/results_groups_" + version + ".npy", allow_pickle=True).item()
    mean_bas = np.load(processed_data_path + "results/results_recon_mean_baseline_" + version + ".npy", allow_pickle=True).item()
    recon_results_1 = np.load(processed_data_path + "results/results_recon_no_corr_" + version + ".npy", allow_pickle=True).item()
    cor_results = np.load(processed_data_path + "wp2.2/sig_overlap/cor_results_" + version + ".npy", allow_pickle=True).item()
    
    ### Starting the analysis
    
    # Getting the reconstruction average results
    recon_average = cal_reconstruction_change(recon_results, repeats)
    
    # Getting overlapping hits
    sig_hits, median_p_val = overlapping_hits(nLatents, cor_results, repeats, con_names, drug)
    
    # Getting high supported hits
    all_hits, collected_overlap = identify_high_supported_hits(sig_hits, drug_h, version, processed_data_path)
    
    # Saving the pi values of results of overlapping_hits() and  identify_high_supported_hits() functions
    report_values(processed_data_path, sig_hits, median_p_val, drug_h, all_hits, collected_overlap, con_names)
    
    # Calculating average change among different runs
    con_list_concat = np.concatenate(con_list, axis=-1)
    recon_average_corr_new_all, recon_average_corr_all_indi_new = get_change_in_reconstruction(recon_average, groups, drug, drug_h, con_names, collected_overlap, sig_hits, con_list_concat, version, processed_data_path, types)
    
    # Loading the results saved by get_change_in_reconstruction() - for using the results without the need to rerun the function
    recon_average_corr_new_all = np.load(processed_data_path + "results/results_confidence_recon_all_" + version + ".npy", allow_pickle=True)
    recon_average_corr_all_indi_new = np.load(processed_data_path + "results/results_confidence_recon_all_indi_" + version + ".npy", allow_pickle=True).item()
    
    # Writing all the hits for each drug and database separately. Also, writing what features were increased or decreased with the association with the drug 
    write_omics_results(processed_data_path, up_down_list, collected_overlap, recon_average_corr_new_all, headers_all, continuous_names, drug_h, con_names)
    
    # Saving the effect sizes (95 % interval) of results of get_change_in_reconstruction() functions
    make_files(collected_overlap, groups, con_list_concat, processed_data_path, recon_average_corr_all_indi_new, con_names, continuous_names, drug_h, drug, all_hits, types, version)
    
    # Getting inter drug variation 
    df_indi_var = get_inter_drug_variation(con_names, drug_h, recon_average_corr_all_indi_new, groups, collected_overlap, drug, con_list_concat, processed_data_path, types)

    # Visualizing variation, heatmap of similarities within drugs across all data and specific for each omics
    visualize_indi_var(df_indi_var, version, processed_data_path)
    visualize_drug_similarity_across_all(recon_average_corr_new_all, drug_h, version, processed_data_path)
    get_drug_similar_each_omics(con_names, continuous_names, all_hits, recon_average_corr_new_all, drug_h, version, processed_data_path)
    
if __name__ == "__main__":
    main()